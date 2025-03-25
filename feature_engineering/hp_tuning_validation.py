import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

class HPTuningValidation:
    def __init__(self):
        self.X, self.y = self._load_data()

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the data from the csv file and return the X and y dataframes 
        Returns:
            tuple: X and y dataframes
        """
        print('loading data...')

        data = pd.read_csv('covtype.data.gz', compression='gzip', header=None)
        # 55th column is target cover_type
        return data.iloc[:, :-1], data.iloc[:, -1]

    def _preprocess_data(self, style: str = "matrix", subset: int = 0.3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Preprocess the data by removing highly correlated columns and return the train and test data 
        Parameters:
            style (str): The style of preprocessing to be done. Either 'matrix' or 'pca'
            subset (int): The percentage of data to be used for training and testing to speed up processing. Default is 0.3
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print('preprocessing data...')

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=subset, stratify=self.y)
        
        if style == "matrix":
            # correlation matrix to remove highly correlated columns
            corr = X_train.corr()
            for cor in corr.columns:
                # dropping columns with correlation > 0.8, iloc[1] because iloc[0] is the column itself 
                if corr[cor].sort_values(ascending=False).iloc[1] > 0.8:
                    X_train.drop(cor, axis=1, inplace=True)
                    X_test.drop(cor, axis=1, inplace=True)
        elif style == "pca":
            pca = PCA(n_components=0.8)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        else:
            raise ValueError("style must be either 'matrix' or 'pca'")

        return X_train, X_test, y_train, y_test
    
    def fit_predict(self) -> None:
        """Fit the model and predict the accuracy of the model"""
        X_train, X_test, y_train, y_test = self._preprocess_data(style="matrix")

        models = [GradientBoostingClassifier(), MLPClassifier()]
        params = [
            {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0],
            },
            {
            'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            }
        ]
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        
        all_models = []

        for model, param in zip(models, params):
            print(f'Random Searching {model.__class__.__name__}...')
            random_search = RandomizedSearchCV(model, param_distributions=param, cv=skf, n_iter=3, verbose=3, n_jobs=-1)
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            accuracy = best_model.score(X_test, y_test)

            all_models.append({
                'model': best_model,
                'params': best_params,
                'accuracy': accuracy
            })

            print(f'Best Model from Grid Search: {best_model.__class__.__name__}')
            print(f'Accuracy From Grid Search: {accuracy}')
            print(f'Best params: {best_params}')

        best_model = max(all_models, key=lambda x: x['accuracy'])
        print(f'Best Model: {best_model["model"].__class__.__name__}')
        print(f'Testing Accuracy: {best_model["accuracy"]}')
        print(f'Best params: {best_model["params"]}')

        best_model['model'].fit(X_train, y_train)
        y_pred = best_model['model'].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Final Accuracy: {accuracy}')

if __name__ == '__main__':
    hp = HPTuningValidation()
    hp.fit_predict()