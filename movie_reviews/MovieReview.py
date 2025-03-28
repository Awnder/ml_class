import requests
import tarfile
import os
import re
import string
import pandas as pd
from BagOfWords import BagOfWords
import tensorflow as tf

class MovieReview:
    def __init__(self):
        self.bag_of_words = BagOfWords(extra_stopwords=["movie", "film", "br", "one"])
        self.model = None

    def fit(self, use_saved_model: bool = True) -> None:
        """Fits the model on the training data"""
        if not os.path.exists("train_pos.csv") or not os.path.exists("train_neg.csv"):
            self._download_imdb_data()
            self._create_imdb_csv()
            self._create_bag_of_words()

        if use_saved_model and os.path.exists("trained_model.keras"):
            self.model = tf.keras.models.load_model("trained_model.keras")
            return

        train_data = pd.concat(
            [
                pd.read_csv("train_pos.csv", encoding="utf-8", header=0),
                pd.read_csv("train_neg.csv", encoding="utf-8", header=0)
            ]
        )
        
        train_target_tensor = tf.convert_to_tensor(train_data["target"].values, dtype=tf.int32)

        train_content_tensor = self.bag_of_words.bag(train_data["content"].values)

        print(train_content_tensor)
        return
        # vectorize_layer = tf.keras.layers.TextVectorization(
        #     max_tokens=1000,
        #     output_mode='int',
        #     output_sequence_length=30
        # )
        # vectorize_layer.adapt(train_data['content'].values)
        # train_content_tensor = vectorize_layer(train_data['content'].values)
        # train_dataset = tf.data.Dataset.from_tensor_slices((train_content_tensor, train_target_tensor))
        # train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).batch(32)

        model = tf.keras.Sequential(
            [
            tf.keras.layers.Embedding(
                input_dim=vectorize_layer.vocabulary_size(),
                output_dim=128,
                mask_zero=True,
            ),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(11, activation="softmax"),
            ]
        )

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        model.fit(
            train_dataset,
            epochs=10,
            batch_size=32,
            shuffle=True,
            verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
        )
        model.save("trained_model.keras")
        self.model = model

    def predict(self, text: str) -> int:
        """Predicts the sentiment of the given text
        Args:
            text (str): The text to predict the sentiment for
        Returns:
            int: The predicted sentiment score
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        bag_of_words = self.bag_of_words.bag(text)

        prediction = self.model.predict(bag_of_words)
        return prediction

    def _create_bag_of_words(self) -> None:
        """Creates a bag of words from the training data
        Args:
            percentage (float): Percentage of the training data to use for creating the bag of words
        """
        if os.path.exists("train_pos.csv"):
            df = pd.read_csv("train_pos.csv", encoding="utf-8")
            self.bag_of_words.adapt(df["content"])

        if os.path.exists("train_neg.csv"):
            df = pd.read_csv("train_neg.csv", encoding="utf-8")
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
    movie_review.fit()
    # p = movie_review.predict("This movie was fantastic! I loved it.")
    # print(p)
    # print(movie_review.bag_of_words.empty())  # Should return an empty dictionary
