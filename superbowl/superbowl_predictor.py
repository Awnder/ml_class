from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import http.client
from dotenv import load_dotenv
import os
import json
import df_nfl_team_statistics
import df_nfl_team_listing

# predict score for eagles and chiefs - 2 predictions

# (Get NFL Team Detail by Team ID) team listing gives team ids
# (Get NFL Team Statistics) nfl team stats gives team stats for a season using id (same as Get NFL Team Stats by Team ID)
# (GET NFL Scoreboard by Year) all ids for games under events
# (GET NFL Event/Game Detail Info) gets game info by game id, competitions->competitors gives win/loser as team id or displayname, winner: bool

### API REQUEST FUNCTIONS
def retrieve_api_data(url: str, restapi: str) -> dict:
    """
    Wrapper function to retrieve data from a REST API. Will load data if url matches filename in current directory. Otherwise, will make a request to the REST API and save data as json file.
    Parameters:
        url (str): URL to make a request to
        restapi (str): REST API URL
        data_directory (str): directory to save data. leave blank to use current directory
        enable_rest_request (bool): disables REST request as precaution to avoid unnecessary costs
    Returns:
        dict: json data from REST API or file. Returns None if file doesn't exist or incorrect api url request
    """
    url_cleaned = url
    if url_cleaned[0] == '/':
        url_cleaned = url_cleaned[1:] # requests have a '/' at the beginning, need to remove bc files don't start with /
    url_cleaned = url_cleaned.replace('/', '_') # replace '/' with '_' to match file format (otherwise will create folders)
    url_cleaned = url_cleaned.replace('?', '_') # replace '?' with '_' to match file format 
    url_cleaned = url_cleaned.replace('=', '_') # replace '=' with '_' to match file format

    # search for existing data
    dir_files = os.listdir('api_data')

    for file in dir_files:
        if file == f'{url_cleaned}.json':
            return _load_json_data(f'api_data/{url_cleaned}.json')

    # if don't find, then request
    data = _rest_request(url, restapi)
    _save_json_data(f'api_data/{url_cleaned}.json', data)
    return _load_json_data(f'api_data/{url_cleaned}.json')

def _rest_request(url: str, restapi: str) -> str:
    """ Makes a request to the REST API """
    conn = http.client.HTTPSConnection(restapi)

    load_dotenv()
    
    headers = {
        'x-rapidapi-key': os.getenv('RAPID_NFL_API_KEY'),
        'x-rapidapi-host': os.getenv('RAPID_NFL_API_HOST')
    }

    conn.request("GET", url, headers=headers)

    res = conn.getresponse()
    data = res.read()

    return data.decode('utf-8')

### DATA PROCESSING FUNCTIONS
def _save_json_data(filename: str, data: str) -> None:
    """ Takes a desired filename and json string data and dumps into file """
    with open(filename, 'w') as fout:
        json_string_data = json.dumps(data)
        fout.write(json_string_data)
        
def _load_json_data(filename: str) -> dict:
    """ Takes a filename and loads json data from file """
    json_data = None
    with open(filename) as fin:
        json_data = json.load(fin)

    json_data = json.loads(json_data)
    return json_data

def load_team_data_to_dataframe(TEAM_IDS: list, TEAM_NAMES: list, year_range: list[int], stats_to_use: list[str]) -> pd.DataFrame:
    """ Using the team ids and names, load the team data into a dataframe, """
    total_df = pd.DataFrame()

    for year in range(year_range[0], year_range[1]+1):
        for team_id, team_name in zip(TEAM_IDS, TEAM_NAMES):
            df = df_nfl_team_statistics.json_to_dataframe(f'api_data/nfl-team-statistics_id_{team_id}&year_{year}.json')
            df = df[['stat_name', 'stat_value']].drop_duplicates(subset=['stat_name']) # isolate the stat columns
            df = df.T # transpose so features are now columns
            df.columns = df.iloc[0] # set first row as column names
            df = df.drop(df.index[0])
            df = df.loc[:, stats_to_use] # only keep the features we want
            df['team'] = team_name

            total_df = pd.concat([total_df, df], axis=0)

    total_df.reset_index(drop=True, inplace=True)
    return total_df

### LINEAR REGRESSION FUNCTIONS
def scaler_normalize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Normalize data using MinMaxScaler, returns normalized X_train and X_test """
    mmscalar = MinMaxScaler()
    X_train_encoded = mmscalar.fit_transform(X_train.drop('team', axis=1))
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=X_train.drop('team', axis=1).columns)
    X_test_encoded = mmscalar.transform(X_test.drop('team', axis=1))
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=X_test.drop('team', axis=1).columns)
    return mmscalar, X_train_encoded, X_test_encoded

def hotencoder_normalize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Normalize names using one hot encoding """
    ohencoder = OneHotEncoder()
    X_train_names_encoded = ohencoder.fit_transform(X_train['team'].to_numpy().reshape(-1, 1))
    X_train_names_encoded = pd.DataFrame(X_train_names_encoded.toarray(), columns=ohencoder.get_feature_names_out())
    X_test_names_encoded = ohencoder.transform(X_test['team'].to_numpy().reshape(-1, 1))
    X_test_names_encoded = pd.DataFrame(X_test_names_encoded.toarray(), columns=ohencoder.get_feature_names_out())
    return ohencoder, X_train_names_encoded, X_test_names_encoded

def train_linear_regression_model(total_df: pd.DataFrame, test: bool=False) -> tuple[LinearRegression, OneHotEncoder, MinMaxScaler]:
    """ Test linear regression model, if test is True, will split data into train and test sets """
    X_data = total_df.drop('totalPointsPerGame', axis=1)
    y_data = total_df[['totalPointsPerGame']]

    if test:
        print('testing linear regression model')
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

        ohencoder, X_train_names_encoded, X_test_names_encoded = hotencoder_normalize_data(X_train, X_test)
        mmscaler, X_train_encoded, X_test_encoded = scaler_normalize_data(X_train, X_test)

        X_train = pd.concat([X_train_encoded, X_train_names_encoded], axis=1)
        X_test = pd.concat([X_test_encoded, X_test_names_encoded], axis=1)
        X_train.sort_index(axis=1, inplace=True)
        X_test.sort_index(axis=1, inplace=True)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        for i in range(len(y_pred)):
            print('Predicted:', y_pred[i], 'Actual:', y_test.iloc[i].values[0])
          
        print('Model Score:', mean_squared_error(y_test, y_pred))
        
        return model, ohencoder, mmscaler
    else:
        print('training linear regression model and returning it for predictions')
        ohencoder, X_train_names_encoded, _ = hotencoder_normalize_data(X_data, X_data)
        mmscaler, X_train_encoded, _ = scaler_normalize_data(X_data, X_data)

        X_train = pd.concat([X_train_encoded, X_train_names_encoded], axis=1)
        X_train.sort_index(axis=1, inplace=True)

        model = LinearRegression()
        model.fit(X_train, y_data)

        return model, ohencoder, mmscaler
        

def main():
    restapi = 'api-nfl-v1.p.rapidapi.com'

    # getting team ids and names
    retrieve_api_data('/nfl-team-listing/v1/data', restapi)
    team_ids_df = df_nfl_team_listing.json_to_dataframe('api_data/nfl-team-listing_v1_data.json')
    TEAM_IDS = team_ids_df['id'].tolist()
    TEAM_NAMES = team_ids_df['displayName'].tolist()
    PREDICTION_NAMES = ['Philadelphia Eagles', 'Kansas City Chiefs']
    PREDICTION_IDS = team_ids_df[team_ids_df['displayName'].isin(PREDICTION_NAMES)]['id'].tolist()

    # seasons go back to 1922 - but unsure of data quality - sticking with 10 years of data
    YEARS = [2014, 2023]
    STATS = ['avgInterceptionYards', 'avgSackYards', 'turnOverDifferential', 'avgStuffYards', 'avgGain', 'possessionTimeSeconds', 'yardsPerGame', 'totalPointsPerGame']
    # STATS = ['totalYards', 'totalPenaltyYards', 'sackYardsLost', 'turnOverDifferential', 'totalGiveaways', 'totalTakeaways', 'possessionTimeSeconds', 'yardsPerGame', 'totalPointsPerGame']

    print('retrieving team data')
    for year in range(YEARS[0], YEARS[1]+1):
        for team_id in TEAM_IDS:
            get_route = f'/nfl-team-statistics?id={team_id}&year={year}'
            retrieve_api_data(get_route, restapi)

    print('loading team data into dataframe')
    total_df = load_team_data_to_dataframe(TEAM_IDS, TEAM_NAMES, YEARS, STATS)

    ### Linear Regression
    model, ohencoder, mmscaler = train_linear_regression_model(total_df, True)

    ohencoder.transform(total_df['team'].to_numpy().reshape(-1, 1))
    other_team_names = ohencoder.get_feature_names_out()

    ### actual prediction
    for team_id, team_name in zip(PREDICTION_IDS, PREDICTION_NAMES):
        # get current team data
        get_route = f'/nfl-team-statistics?id={team_id}&year={2024}' # 2024 is current year for superbowl predictions
        retrieve_api_data(get_route, restapi)
        current_team_df = load_team_data_to_dataframe([team_id], [team_name], [2024, 2024], STATS)
        current_team_df.drop('totalPointsPerGame', axis=1, inplace=True)

        # normalize data using previously fitted MinMaxScaler and OneHotEncoder 
        X_train_encoded = mmscaler.transform(current_team_df.drop('team', axis=1))
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=current_team_df.drop('team', axis=1).columns)
        X_train_names_encoded = ohencoder.transform(current_team_df['team'].to_numpy().reshape(-1, 1))
        X_train_names_encoded = pd.DataFrame(X_train_names_encoded.toarray(), columns=ohencoder.get_feature_names_out())

        X_data = pd.concat([X_train_encoded, X_train_names_encoded], axis=1)

        for col in other_team_names:
            if col not in X_data.columns:
                X_data[col] = 0

        X_data.sort_index(axis=1, inplace=True) # data must be in same order as fit, sorting alphabetically
        print(X_data)
        # predict!
        prediction_score = model.predict(X_data).astype(int)
        print(f'{PREDICTION_NAMES[PREDICTION_IDS.index(team_id)]} predicted score:', prediction_score)     


if __name__ == '__main__':
    main()
  