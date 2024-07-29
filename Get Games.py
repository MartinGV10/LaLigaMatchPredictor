import requests
from bs4 import BeautifulSoup
import pandas as pd

url = requests.get('https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures').text
soup = BeautifulSoup(url, 'lxml')
table = soup.select('table.stats_table')[0]

games_table = pd.read_html(url, match='Scores & Fixtures ')[0]

sorted = games_table.sort_values('Home')

print(sorted)

sorted.to_csv('results.csv')
