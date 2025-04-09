import requests
import tarfile
import os
import re
import string
import pandas as pd
from BagOfWords import BagOfWords
import torch

class MovieReviewNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """Initialized Movie Review Neural Network
        Args:
            input_dim (int): The size of the input vocabulary
            hidden_dim (int, optional): The number of hidden units in the LSTM layers. Defaults to 128.
            output_dim (int, optional): The number of output classes. Defaults to 1 for binary classification.
        """
        super(MovieReviewNN, self).__init__()
        self.embedding = torch.nn.Embedding(input_dim, 128)  # Embedding layer for word indices
        self.lstm1 = torch.nn.LSTM(128, hidden_dim) 
        self.lstm2 = torch.nn.LSTM(hidden_dim, hidden_dim // 2)
        self.fc = torch.nn.Linear(hidden_dim // 2, 1)  # binary output layer 0/neg or 1/pos
        self.sigmoid = torch.nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, (hn, cn) = self.lstm2(x)
        x = self.fc(hn[-1])
        x = self.sigmoid(x)
        return x
    
class MovieReview:
    def __init__(self):
        """Initialized Movie Review Class with a Bag of Words and a neural network"""
        self.bag_of_words = BagOfWords(vocabulary_size=1000, extra_stopwords=["movie", "film", "br", "one"])
        self.model = None

    def fit(self, train_loader: torch.utils.data.DataLoader, num_epochs: int = 10, use_saved_model: bool = True) -> None:
        """Fits the model on the training data
        Args:
            train_loader (DataLoader): DataLoader for the training data
            num_epochs (int): Number of epochs to train the model
            use_saved_model (bool): Whether to use a saved model if available
        """
        # Check if a saved model exists and load it
        if use_saved_model and os.path.exists("trained_model.pth"):
            self.model = MovieReviewNN(input_dim=self.bag_of_words.vocabulary_size)
            self.model.load_state_dict(torch.load("trained_model.pth"))
            return
        else:
            self.model = MovieReviewNN(input_dim=self.bag_of_words.vocabulary_size)
            self.model.train()

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)  # Optimizer for training

        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(), targets.float())
                running_loss += loss.item() * inputs.size(0)

                # Backward pass and optimization
                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {running_loss / len(train_loader):.2f}")  # Print loss for each batch

        torch.save(self.model.state_dict(), "trained_model.pth")  # Save the trained model state

    def predict_text(self, test_text: str) -> int:
        """Predicts the sentiment of a single text input
        Args:
            test_text (str): The text to predict sentiment for
        Returns:
            int: The predicted sentiment class 0 or 1
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            bag_of_words = self.bag_of_words.bag([test_text])
            print('bag_of_words',bag_of_words)
            bag_of_words = torch.tensor(bag_of_words[0])
            print('bag_of_words tensor',bag_of_words)
            outputs = self.model(bag_of_words)
            print('outputs',outputs)
            outputs = (outputs > 0.5).float()  # Convert probabilities to binary class labels
            print('binary class',outputs)
            predicted_class = outputs.item()
            print(f"Predicted class: {predicted_class}")  # Print the predicted class
        
        return predicted_class

    def predict_loader(self, test_loader: torch.utils.data.DataLoader) -> None:
        """Measures model accuracy using test data
        Args:
            test_loader (DataLoader): DataLoader containing the test data
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        self.model.eval()

        correct = 0
        total = 0
        for inputs, targets in test_loader:
            outputs = self.model(inputs)
            predicted = torch.argmax(outputs, dim=1).unsqueeze(0).float()
            true_labels = torch.argmax(targets, dim=0)
            print(f"Predicted: {predicted.tolist()}")
            print(f"True Labels: {true_labels.tolist()}")
            total += true_labels
            correct += (predicted == true_labels).sum().item()

        accuracy = correct / total
        print(f"Accuracy: {accuracy:.2f}%")

    def preprocess_data(self) -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
        """Preprocesses the data for training and testing
        Returns:
            tuple: A tuple containing the train and test datasets
        """
        if not os.path.exists("train_pos.csv") or not os.path.exists("train_neg.csv"):
            self._download_imdb_data()
            self._create_imdb_csv()

        # Load the data from CSV files
        train_df = pd.concat(
            [
                pd.read_csv("train_pos.csv", encoding="utf-8"), 
                pd.read_csv("train_neg.csv", encoding="utf-8")
            ], 
            ignore_index=True
        )
        test_df = pd.concat(
            [
                pd.read_csv("test_pos.csv", encoding="utf-8"), 
                pd.read_csv("test_neg.csv", encoding="utf-8")
            ], 
            ignore_index=True
        )

        self.create_bag_of_words()

        # set target to 0 or 1 for positive or negative review
        train_df["target"] = train_df["target"].apply(lambda x: 1 if x >= 5 else 0)
        test_df["target"] = test_df["target"].apply(lambda x: 1 if x >= 5 else 0)

        # Convert to tensor datasets
        X_train = torch.tensor(self.bag_of_words.bag(train_df["content"].tolist()))
        y_train = torch.tensor(train_df["target"].values)
        X_test = torch.tensor(self.bag_of_words.bag(test_df["content"].tolist()))
        y_test = torch.tensor(test_df["target"].values)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        return train_dataset, test_dataset

    def create_bag_of_words(self) -> None:
        """Creates a bag of words from the training data
        Args:
            percentage (float): Percentage of the training data to use for creating the bag of words
        """
        df = None
        if os.path.exists("train_pos.csv") and os.path.exists("train_neg.csv"):
            df = pd.concat(
                [
                    pd.read_csv("train_pos.csv", encoding="utf-8"), 
                    pd.read_csv("train_neg.csv", encoding="utf-8")
                ],
                ignore_index=True
            )

        self.bag_of_words.adapt(df["content"])

    def _download_imdb_data(self, dir_dest_path: str = "aclImdb") -> None:
        """Downloads and extracts Imdb data
        Args:
            dir_dest_path (str): Destination path for the extracted data
        """
        if not os.path.exists("aclImdb_v1.tar.gz"):
            response = requests.get(
                "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            )
            if response.status_code == 200:
                with open("aclImdb_v1.tar.gz", "wb") as f:
                    f.write(response.content)
            else:
                print("Failed to download data.")
                return

        if not os.path.exists(dir_dest_path):
            with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
                members = [
                    member
                    for member in tar.getmembers()
                    if member.name.startswith(f"{dir_dest_path}/train/")
                    or member.name.startswith(f"{dir_dest_path}/test/")
                    or member.name.startswith(f"{dir_dest_path}/README")
                ]
                tar.extractall(members=members)

    def _create_imdb_csv(self, dir_source_path: str = "aclImdb") -> None:
        """Processes the Imdb data into train and test CSV files
        Args:
            dir_source_path (str): Source path for the extracted data
        """
        if not os.path.exists("train_pos.csv"):
            train_pos_files = os.listdir(
                os.path.join(f"{dir_source_path}", "train", "pos")
            )

            with open("train_pos.csv", "w", encoding="utf-8") as out_f:
                out_f.write("target,content\n")

            for file in train_pos_files:
                source_train_pos_file = os.path.join(
                    f"{dir_source_path}", "train", "pos", file
                )
                with open(source_train_pos_file, "r", encoding="utf-8") as in_f:
                    with open("train_pos.csv", "a", encoding="utf-8") as out_f:
                        target = file[:-4].split("_")[1]
                        content = in_f.read()
                        out_f.write(f"{target},{self._clean_text(content)}\n")

        if not os.path.exists("train_neg.csv"):
            train_neg_files = os.listdir(
                os.path.join(f"{dir_source_path}", "train", "neg")
            )

            with open("train_neg.csv", "w", encoding="utf-8") as out_f:
                out_f.write("target,content\n")

            for file in train_neg_files:
                source_train_neg_file = os.path.join(
                    f"{dir_source_path}", "train", "neg", file
                )
                with open(source_train_neg_file, "r", encoding="utf-8") as in_f:
                    with open("train_neg.csv", "a", encoding="utf-8") as out_f:
                        target = file[:-4].split("_")[1]
                        content = in_f.read()
                        out_f.write(f"{target},{self._clean_text(content)}\n")

        if not os.path.exists("test_pos.csv"):
            test_pos_files = os.listdir(
                os.path.join(f"{dir_source_path}", "test", "pos")
            )

            with open("test_pos.csv", "w", encoding="utf-8") as out_f:
                out_f.write("target,content\n")

            for file in test_pos_files:
                source_test_pos_file = os.path.join(
                    f"{dir_source_path}", "test", "pos", file
                )
                with open(source_test_pos_file, "r", encoding="utf-8") as in_f:
                    with open("test_pos.csv", "a", encoding="utf-8") as out_f:
                        target = file[:-4].split("_")[1]
                        content = in_f.read()
                        out_f.write(f"{target},{self._clean_text(content)}\n")

        if not os.path.exists("test_neg.csv"):
            test_neg_files = os.listdir(
                os.path.join(f"{dir_source_path}", "test", "neg")
            )

            with open("test_neg.csv", "w", encoding="utf-8") as out_f:
                out_f.write("target,content\n")

            for file in test_neg_files:
                source_test_neg_file = os.path.join(
                    f"{dir_source_path}", "test", "neg", file
                )
                with open(source_test_neg_file, "r", encoding="utf-8") as in_f:
                    with open("test_neg.csv", "a", encoding="utf-8") as out_f:
                        target = file[:-4].split("_")[1]
                        content = in_f.read()
                        out_f.write(f"{target},{self._clean_text(content)}\n")

    def _clean_text(self, text: str) -> str:
        text = text.replace("\n", " ").replace("\r", " ")  # remove newlines
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )  # remove punctuation
        text = re.compile(r"<[^>]+>").sub("", text)  # remove HTML tags
        text = text.lower()  # convert to lowercase
        return text

if __name__ == "__main__":
    movie_review = MovieReview()
    train_dataset, test_dataset = movie_review.preprocess_data()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, drop_last=True)
    movie_review.fit(train_loader, num_epochs=2)

    # text = "This movie was great! I loved it."
    text = "I hated bad worst. ruin terrible."
    p = movie_review.predict_text(test_text=text)
    print(f"Predicted sentiment for the text: {text} is {p}")
    # movie_review.predict_loader(test_loader)
