import re

file = 'tv_my_little_pony_season1.txt'

with open(file, 'r') as f:
    text = f.read()

    # Find all the names in the text
    names = re.findall(r'[A-Z][a-z]+:', text)

    names = [n.replace(":", "") for n in set(names)]
    
    print(names)