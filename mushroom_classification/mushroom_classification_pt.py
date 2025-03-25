import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class MushroomClassifier:
    def __init__(self):
        self.X, self.y = self._load_data()
        self.model_reports = []

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the mushroom dataset."""
        data = pd.read_csv("mushrooms.csv")
        data.index = np.arange(1, len(data) + 1)

        X = data.drop("labels", axis=1)
        y = data["labels"]

        return X, y
    
    def _preprocess_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
     
        return X_train, X_test, y_train, y_test

    def fit_predict(self) -> None:
        """Fits the model and makes predictions."""
        X_train, X_test, y_train, y_test = self._preprocess_data()

        mcnn = MushroomClassifierNN(input_dim=X_train.shape[1], hidden_dim=100, output_dim=2)
    
if __name__ == "__main__":
    mc = MushroomClassifier()
    mc._load_data()
    mc._preprocess_data()


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