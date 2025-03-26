import numpy as np
import pandas as pd
import torch

class MushroomClassifierNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MushroomClassifierNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class MushroomClassifier:
    def __init__(self):
        self.mcnn = None
        self.X, self.y = self._load_data()

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the mushroom dataset."""
        data = pd.read_csv("mushrooms.csv")
        data.index = np.arange(1, len(data) + 1)

        X = data.drop("labels", axis=1)
        y = data["labels"]

        return X, y
    
    def preprocess_data(self) -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset, int]:
        """Preprocesses the data."""
        X_train = self.X[:6000]
        X_test = self.X[6000:]
        y_train = self.y[:6000]
        y_test = self.y[6000:]

        # tensorflow does not support string values, so need to convert to integers
        # we find unique values in the dataset and map them to integers
        # we then use this mapping to convert the string values to integers
        unique_values = pd.unique(X_train.values.ravel())
        value_to_index = {val: idx for idx, val in enumerate(unique_values)}
        X_train = X_train.map(value_to_index.get)
        X_test = X_test.map(value_to_index.get)

        label_mapping = {'p': 0, 'e': 1}
        y_train = y_train.map(label_mapping)
        y_test = y_test.map(label_mapping)

        # Apply one-hot encoding using the numerical values obtained from the mapping
        X_train = torch.nn.functional.one_hot(torch.tensor(X_train.values), num_classes=len(unique_values))
        X_test = torch.nn.functional.one_hot(torch.tensor(X_test.values), num_classes=len(unique_values))
        y_train = torch.nn.functional.one_hot(torch.tensor(y_train.values), num_classes=2)
        y_test = torch.nn.functional.one_hot(torch.tensor(y_test.values), num_classes=2)

        # create correlation matrix and threshold
        # Reshape X_train and X_test to 2D if they are not already
        X_train = X_train.view(X_train.size(0), -1)
        X_test = X_test.view(X_test.size(0), -1)
        
        # Compute the correlation matrix - corrcoef only takes 2D tensors
        corr_matrix = torch.corrcoef(X_train.T)
        threshold = 0.8

        # identify highly correlated features above threshold
        # the correlation matrix is symmetric, so we only need to look at the upper triangle
        upper_tri = torch.triu(corr_matrix, diagonal=1)
        highly_corr = torch.where(torch.abs(upper_tri) > threshold)
        
        # get the indices of the features to drop
        features_to_drop = []
        for i, j in zip(*highly_corr):
            if i not in features_to_drop:
                features_to_drop.append(j.item())

        # create mask to keep non-dropped features while retaining one of the correlated features
        mask = torch.ones(X_train.shape[1], dtype=torch.bool)
        mask[features_to_drop] = False

        X_train = X_train[:, mask]
        X_test = X_test[:, mask]

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
     
        return train_dataset, test_dataset, X_train.shape[1]
    
    def fit(self, train_loader: torch.utils.data.DataLoader, num_epochs: int = 10) -> None:
        self.mcnn = MushroomClassifierNN(input_dim=input_dim, hidden_dim=100, output_dim=2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.mcnn.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            self.mcnn.train()
            running_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                outputs = self.mcnn(inputs.float())
                loss = criterion(outputs, torch.argmax(targets, dim=1))
                running_loss += loss.item() * inputs.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx + 1) % 100 == 0:
                    epoch_loss = running_loss / len(train_loader.dataset)
                    print(f"Epoch: {epoch+1}, Loss: {epoch_loss:.2f}")
                    running_loss = 0.0

    def predict(self, test_loader: torch.utils.data.DataLoader) -> None:
        self.mcnn.eval()
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            outputs = self.mcnn(inputs.float())
            predicted = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(targets, dim=1)
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()

        accuracy = correct / total
        print(f"Accuracy: {accuracy:.2f}%")

    def fit_predict(self, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, num_epochs: int = 10) -> None:
        """Fits the model and makes predictions."""
        self.fit(train_loader, num_epochs)
        self.predict(test_loader)

if __name__ == "__main__":
    mc = MushroomClassifier()
    train_dataset, test_dataset, input_dim = mc.preprocess_data()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    mc.fit_predict(train_loader, test_loader, num_epochs=10)