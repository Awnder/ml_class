from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from datetime import datetime
import pandas as pd
import argparse

def knn_titanic(X_train: pd.DataFrame, X_test: pd.DataFrame, hyperparameterize: bool = False):
    ### CLEANING DATA ###
    # setting desired columns
    y_train = X_train["Survived"]
    X_train = X_train[["PassengerId", "Pclass", "Sex", "Age", "Fare"]]
    X_test = X_test[["PassengerId", "Pclass", "Sex", "Age", "Fare"]]

    # setting index
    X_train.set_index("PassengerId", inplace=True)
    X_test.set_index("PassengerId", inplace=True)

    # mapping sex to binary
    X_train["Sex"] = X_train["Sex"].map({"male": 0, "female": 1})
    X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})
    
    # testing only the sex column
    X_train.drop(columns=["Pclass", "Age", "Fare"], inplace=True)
    X_test.drop(columns=["Pclass", "Age", "Fare"], inplace=True)
    # fill missing ages using KNN - replaces missing values with mean from nearest neighbors
    # knn_imputer = KNNImputer(n_neighbors=3)
    # X_train["Age"] = knn_imputer.fit_transform(X_train[["Age"]])
    # X_test["Age"] = knn_imputer.transform(X_test[["Age"]])
    
    # fill single missing class 3 fare value using mean from only class 3 - this code groups by class and fills missing values with the mean of that class
    # X_test["Fare"] = X_test.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.mean()))

    # normalize age and fare data
    # scaler = MinMaxScaler()
    # X_train.loc[:, ["Age", "Fare"]] = pd.DataFrame(scaler.fit_transform(X_train[["Age", "Fare"]].to_numpy()), columns=["Age", "Fare"], index=X_train.index)
    # X_test.loc[:, ["Age", "Fare"]] = pd.DataFrame(scaler.transform(X_test[["Age", "Fare"]].to_numpy()), columns=["Age", "Fare"], index=X_test.index)

    ### PREDICTION ###
    if hyperparameterize:
        params = {
            'n_neighbors': range(1, 31),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        search = RandomizedSearchCV(KNeighborsClassifier(), params, cv=5)
        search.fit(X_train, y_train)

        knn = search.best_estimator_

        print("Hyperparameterization complete")
        print("Best parameters:", search.best_params_)
        print("Best score:", search.best_score_)
    else:
        knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=["Survived"], index=X_test.index)
    
    return y_pred

def random_forest_titanic(X_train: pd.DataFrame, X_test: pd.DataFrame, hyperparameterize: bool = False):
    ### CLEANING DATA ###
    y_train = X_train["Survived"]
    X_train = X_train[["PassengerId", "Pclass", "Sex", "Age"]]
    X_test = X_test[["PassengerId", "Pclass", "Sex", "Age"]]

    # setting index
    X_train.set_index("PassengerId", inplace=True)
    X_test.set_index("PassengerId", inplace=True)

    # mapping sex to binary
    X_train["Sex"] = X_train["Sex"].map({"male": 0, "female": 1})
    X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})

    # normalize the pclass
    scaler = MinMaxScaler()
    X_train["Pclass"] = X_train["Pclass"].astype('float64')
    X_test["Pclass"] = X_test["Pclass"].astype('float64')
    X_train.loc[:, ["Pclass"]] = pd.DataFrame(scaler.fit_transform(X_train[["Pclass"]]), columns=["Pclass"], index=X_train.index)
    X_test.loc[:, ["Pclass"]] = pd.DataFrame(scaler.transform(X_test[["Pclass"]]), columns=["Pclass"], index=X_test.index)
    
    ### PREDICTION ###
    if hyperparameterize:
        params = {
            'n_estimators': range(50, 500),
            'max_depth': range(1, 20),
            'min_samples_split': range(2, 20),
            'min_samples_leaf': range(1, 20),
        }
        
        search = RandomizedSearchCV(RandomForestClassifier(), params, cv=5)
        search.fit(X_train, y_train)

        rf = search.best_estimator_

        print("Hyperparameterization complete")
        print("Best parameters:", search.best_params_)
        print("Best score:", search.best_score_)
    else:
        rf = RandomForestClassifier(n_estimators=100)
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=["Survived"], index=X_test.index)
    
    return y_pred

def decision_tree_titanic(X_train: pd.DataFrame, X_test: pd.DataFrame, hyperparameterize: bool = False):
    ### CLEANING DATA ###
    y_train = X_train["Survived"]
    X_train = X_train[["PassengerId", "Pclass", "Sex", "Age"]]
    X_test = X_test[["PassengerId", "Pclass", "Sex", "Age"]]

    # setting index
    X_train.set_index("PassengerId", inplace=True)
    X_test.set_index("PassengerId", inplace=True)

    # mapping sex to binary
    X_train["Sex"] = X_train["Sex"].map({"male": 0, "female": 1})
    X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})

    # normalize the pclass - need to make pclass an int 
    scaler = MinMaxScaler()
    X_train["Pclass"] = X_train["Pclass"].astype('float64')
    X_test["Pclass"] = X_test["Pclass"].astype('float64')
    X_train.loc[:, ["Pclass"]] = pd.DataFrame(scaler.fit_transform(X_train[["Pclass"]]), columns=["Pclass"], index=X_train.index)
    X_test.loc[:, ["Pclass"]] = pd.DataFrame(scaler.transform(X_test[["Pclass"]]), columns=["Pclass"], index=X_test.index)


    ### PREDICTION ###
    if hyperparameterize:
        params = {
            'max_depth': range(5, 50),
            'min_samples_split': range(2, 20),
            'min_samples_leaf': range(1, 15),
            'max_features': range(1, 15),
            'max_leaf_nodes': range(10, 200),
            'min_impurity_decrease': [i * 0.1 for i in range(0, 5)],
        }
        
        search = RandomizedSearchCV(DecisionTreeClassifier(), params, cv=5)
        search.fit(X_train, y_train)

        dt = search.best_estimator_

        print("Hyperparameterization complete")
        print("Best parameters:", search.best_params_)
        print("Best score:", search.best_score_)
    else:
        dt = RandomForestClassifier(n_estimators=100)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=["Survived"], index=X_test.index)

    return y_pred

def main():
    ### COMMAND LINE INPUT ###
    parser = argparse.ArgumentParser(description="Titanic survival prediction using csv data")
    parser.add_argument("--type", type=str, help="Type of model to use", choices=["knn", "random_forest", "decision_tree"])
    parser.add_argument("--hp", type=bool, help="Whether to hyperparameterize the model", default=False)
    parser.add_argument("--train", type=str, help="Path to train data")
    parser.add_argument("--test", type=str, help="Path to test data")
    args = parser.parse_args()
    
    if not args.train or not args.test:
        parser.print_help()
        return
    
     ### IMPORT DATA ###
    X_train = pd.read_csv(args.train)
    X_test = pd.read_csv(args.test)

    ### PREDICTION ###
    if args.type == "knn":
        y_pred = knn_titanic(X_train, X_test, hyperparameterize=True)
    elif args.type == "random_forest":
        y_pred = random_forest_titanic(X_train, X_test, hyperparameterize=True)
    elif args.type == "decision_tree":
        y_pred = decision_tree_titanic(X_train, X_test, hyperparameterize=True)

    ### OUTPUT ###
    print("Saved to csv:", "titanic_predictions_with_knn.csv")
    y_pred.to_csv(f"titanic_predictions_{args.type}_{datetime.now().minute}{datetime.now().second}.csv")

if __name__ == "__main__":
    main()