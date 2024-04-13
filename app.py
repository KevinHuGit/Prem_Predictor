import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import streamlit as st


st.title('Premier League Match Predictor')


# Web Scraping
st.subheader('Scrape Premier League Seasons')

start = int(st.text_input("Start Date", 2022))
end = int(st.text_input("End Date", 2024))

years = list(range(end,start, -1))

def scrapeGames(start, end):
  all_matches = []
  standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

  for year in years:
      data = requests.get(standings_url)
      soup = BeautifulSoup(data.text)
      standings_table = soup.select('table.stats_table')[0]
      
      links = standings_table.find_all('a')
      links = [l.get("href") for l in links]
      links = [l for l in links if '/squads' in l]
      team_urls = [f"https://fbref.com{l}" for l in links]

      previous_season = soup.select("a.prev")[0].get("href")
      standings_url = f"https://fbref.com{previous_season}"
      
      for team_url in team_urls:
          team_name = team_url.split('/')[-1].replace("-Stats", "").replace("-", " ")
          
          data = requests.get(team_url)
          matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
          
          soup = BeautifulSoup(data.text)
          links = soup.find_all('a')
          links = [l.get("href") for l in links]
          links = [l for l in links if l and 'all_comps/shooting' in l]
        
          data = requests.get(f"https://fbref.com{links[0]}")
          shooting = pd.read_html(data.text, match="Shooting")[0]
          shooting.columns = shooting.columns.droplevel()
          
          try:
              team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
          except ValueError:
              continue
              
          team_data = team_data[team_data["Comp"] == "Premier League"]
          
          team_data["Season"] = year
          team_data["Team"] = team_name
          all_matches.append(team_data)
          time.sleep(2)
  return all_matches

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Scrape Web', on_click=click_button)

if st.session_state.clicked:
  match_df = pd.concat(scrapeGames(start, end))
  st.write(match_df)



# ML Prediction

st.subheader('Predict Premier League Match Results')

matches = pd.read_csv("matches[long].csv", index_col=0)

matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2024-01-01']
test = matches[matches["date"] > '2024-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])

acc = accuracy_score(test["target"], preds)

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])

st.write("Prediction Accuracy: ")
st.write(precision_score(test["target"], preds))

grouped_matches = matches.groupby("team")

user_input = st.text_input('Enter a Team', 'Arsenal')
group = grouped_matches.get_group(user_input).sort_values("date")
st.write(group)