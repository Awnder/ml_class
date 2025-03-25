import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA


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
    
    def _preprocess_data(self, style="matrix") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Preprocesses the data."""
        X_train = self.X[:6000]
        X_test = self.X[6000:]
        y_train = self.y[:6000]
        y_test = self.y[6000:]

        enc = TargetEncoder()
        X_train = pd.DataFrame(enc.fit_transform(X_train, y_train), columns=self.X.columns)
        X_test = pd.DataFrame(enc.transform(X_test), columns=self.X.columns)

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=self.X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=self.X.columns)

        if style == "matrix":
            # correlation matrix to remove highly correlated columns
            corr = X_train.corr()
            for cor in corr.columns:
                # dropping columns with correlation > 0.7, iloc[1] because iloc[0] is the column itself 
                if corr[cor].sort_values(ascending=False).iloc[1] > 0.7:
                    X_train.drop(cor, axis=1, inplace=True)
                    X_test.drop(cor, axis=1, inplace=True)
        elif style == "pca":
            pca = PCA(n_components=0.8)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        elif style == "common":
            X_train = X_train[["stalk_root", "stalk_color_above_ring", "stalk_color_below_ring", "gill_color", "cap_color", "odor", "ring_number", "ring_type", "population", "habitat"]]
            X_test = X_test[["stalk_root", "stalk_color_above_ring", "stalk_color_below_ring", "gill_color", "cap_color", "odor", "ring_number", "ring_type", "population", "habitat"]]
        else:
            raise ValueError("style must be either 'matrix' or 'pca' or 'common'")

        return X_train, X_test, y_train, y_test

    def fit_predict(self, style="matrix") -> None:
        """Trains RandomForest, KNeighbors, and LogisticRegression models."""
        self.model_reports = []
        X_train, X_test, y_train, y_test = self._preprocess_data(style)
        
        rf = RandomForestClassifier(n_estimators=20)
        rf.fit(X_train, y_train)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        mlp = MLPClassifier(hidden_layer_sizes=(20, 10))
        mlp.fit(X_train, y_train)

        for model in [rf, knn, lr, mlp]:
            self.model_reports.append(classification_report(y_test, model.predict(X_test), output_dict=True))

    def best_recall(self) -> str:
        """Prints the model and classification report with the best recall to predict poisonous mushrooms."""
        recalls = [report["p"]["recall"] for report in self.model_reports]
        best_model = ["RandomForest", "KNeighbors", "LogisticRegression", "MLPClassifier"][recalls.index(max(recalls))]

        print(f"Best model: {best_model}")
        print(f"Recall: {max(recalls)}")
        
    def best_precision(self) -> str:
        """Prints the model and classification report with the best precision to predict poisonous mushrooms."""
        precisions = [report["p"]["precision"] for report in self.model_reports]
        best_model = ["RandomForest", "KNeighbors", "LogisticRegression", "MLPClassifier"][precisions.index(max(precisions))]

        print(f"Best model: {best_model}")
        print(f"Precision: {max(precisions)}")

    
if __name__ == "__main__":
    mc = MushroomClassifier()

    print('-----Feature Reducing with Matrix-----')
    mc.fit_predict(style="matrix")
    mc.best_recall()

    print('-----Feature Reducing with PCA-----')
    mc.fit_predict(style="pca")
    mc.best_recall()

    print('-----Feature Reducing with Common-----')
    mc.fit_predict(style="common")
    mc.best_recall()