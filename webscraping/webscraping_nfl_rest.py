import http.client
import os
from dotenv import load_dotenv
import json

url = "/nfl-team-statistics?year=2023&id=22"
restapi = "nfl-api-data.p.rapidapi.com"
keys = [os.getenv("RAPID_NFL_API_KEY"), os.getenv("RAPID_NFL_API_HOST")]

def retrieve_api_data(url: str, restapi: str, keys: list, enable_search=False):
    """
    Wrapper function to retrieve data from a REST API. Will load data if url matches filename in current directory. Otherwise, will make a request to the REST API and save data as json file.
    Parameters:
        url (str): URL to make a request to
        restapi (str): REST API URL
        keys (list): keys to access REST API like [API_KEY, API_HOST]
        enable_search (bool): disables REST request as precaution to avoid unnecessary costs
    Returns:
        dict: json data from REST API or file. Returns dict with 'content' key if file doesn't exist or incorrect api url request
    """
    data = rest_request(url, restapi, keys)

    current_dir_files = os.listdir(os.getcwd())

    for file in current_dir_files:
        if file == f'{url}.json':
            return load_json_data(f'{url}.json')

    if enable_search:
        save_json_data(f'{url}.json', data)
        return {'content': 'Data saved to json file'}

def rest_request(url: str, restapi: str, keys: list):
	conn = http.client.HTTPSConnection(restapi)

	load_dotenv()
	
	headers = {
		'x-rapidapi-key': keys[0],
		'x-rapidapi-host': keys[1]
	}

	conn.request("GET", url, headers=headers)

	res = conn.getresponse()
	data = res.read()

	return data.decode('utf-8')

def save_json_data(filename, data):
    with open(filename, 'w') as fout:
        json_string_data = json.dumps(data)
        fout.write(json_string_data)
        
def load_json_data(filename):
    json_data = None
    with open(filename) as fin:
        json_data = json.load(fin)

    json_data = json.loads(json_data)
    return json_data


def test():
    # data = rest_request() # NOTE THIS WILL COST
    # save_json_data('nfl_team_stats.json', data)

    jdata = load_json_data('nfl_team_stats.json')

    ### PARSING JSON DATA AND FINDING NECESSARY KEYS

    # CATEGORIES
    categories = jdata['statistics']['splits']['categories'] # a list of categories for each team
    catkeys = categories[0].keys() # dictionary of valid keys
    catkeys = ['stats'] # WANTED KEYS

    # STATS FOR EACH CATEGORY
    # for i in range(len(categories[0]['stats'])):
    #     print(f'category: {categories[0]["stats"][i]}')
    #     print('------------------------------')

    catstatskeys = ['displayName', 'displayValue', 'perGameDisplayValue', 'rankDisplayValue', 'description'] # WANTED KEYS

    # TESTING STAT SEARCH USING ONLY ONE CATEGORY
    # for i in range(len(categories[0]['stats'])):
    #     for catstatkey in catstatskeys:
    #         if catstatkey in categories[0]['stats'][i]:
    #             print(f'{catstatkey}: {categories[0]["stats"][i][catstatkey]}')
    #             print('------------------------------')

    # WRITE TO CSV
    with open('nfl_team_stats_parsed.csv', 'w') as fout:
        fout.write(','.join(catstatskeys) + '\n')
        for team in range(len(categories)):
            for stat in range(len(categories[team]['stats'])):
                for catstatkey in catstatskeys:
                    if catstatkey in categories[team]['stats'][stat]:
                        fout.write(f'{categories[team]["stats"][stat][catstatkey]},')
                fout.write('\n')


    # USE THIS TO SEE PRINT OUT
    # for team in range(len(categories)): # 10 teams
    #     for stat in range(len(categories[team]['stats'])):
    #         for catstatkey in catstatskeys:
    #             if catstatkey in categories[team]['stats'][stat]:
    #                 print(f'{catstatkey}: {categories[team]["stats"][stat][catstatkey]}')
    #                 print('------------------------------')