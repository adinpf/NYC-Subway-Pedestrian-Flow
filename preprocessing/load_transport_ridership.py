import pandas as pd
import datetime as dt

df = pd.read_csv("data/transport_ridership.csv")
rename_mapping = {
    'Date': 'date',
    'Subways: Total Estimated Ridership': 'subways_ridership',
    'Subways: % of Comparable Pre-Pandemic Day': 'subways_percent_of_pre',
    'Buses: Total Estimated Ridership': 'buses_ridership',
    'Buses: % of Comparable Pre-Pandemic Day': 'buses_percent_of_pre',
    'LIRR: Total Estimated Ridership': 'lirr_ridership',
    'LIRR: % of Comparable Pre-Pandemic Day': 'lirr_percent_of_pre',
    'Metro-North: Total Estimated Ridership': 'mn_ridership',
    'Metro-North: % of Comparable Pre-Pandemic Day': 'mn_percent_of_pre',
    'Access-A-Ride: Total Scheduled Trips': 'access_a_ride_trips',
    'Access-A-Ride: % of Comparable Pre-Pandemic Day': 'access_a_ride_percent_of_pre',
    'Bridges and Tunnels: Total Traffic': 'bridges_tunnels_traffic',
    'Bridges and Tunnels: % of Comparable Pre-Pandemic Day': 'bridges_tunnels_percent_of_pre',
    'Staten Island Railway: Total Estimated Ridership': 'sir_ridership',
    'Staten Island Railway: % of Comparable Pre-Pandemic Day': 'sir_percent_of_pre'
}

df.rename(columns=rename_mapping)

df["date"] = pd.to_datetime(df["date"])
df["subways_ridership"] = pd.to_numeric(df["subways_ridership"])
df["subways_percent_of_pre"] = pd.to_numeric(df["subways_percent_of_pre"].str.strip('%')) / 100
df["buses_ridership"] = pd.to_numeric(df["buses_ridership"])
df["buses_percent_of_pre"] = pd.to_numeric(df["buses_percent_of_pre"].str.strip('%')) / 100
df["lirr_ridership"] = pd.to_numeric(df["lirr_ridership"])
df["lirr_percent_of_pre"] = pd.to_numeric(df["lirr_percent_of_pre"].str.strip('%')) / 100
df["mn_ridership"] = pd.to_numeric(df["mn_ridership"])
df["mn_percent_of_pre"] = pd.to_numeric(df["mn_percent_of_pre"].str.strip('%')) / 100
df["access_a_ride_trips"] = pd.to_numeric(df["access_a_ride_trips"])
df["access_a_ride_percent_of_pre"] = pd.to_numeric(df["access_a_ride_percent_of_pre"].str.strip('%')) / 100
df["bridge_tunnels_traffic"] = pd.to_numeric(df["bridge_tunnels_traffic"])
df["bridges_tunnels_percent_of_pre"] = pd.to_numeric(df["bridges_tunnels_percent_of_pre"].str.strip('%')) / 100
df["sir_ridership"] = pd.to_numeric(df["sir_ridership"])
df["sir_percent_of_pre"] = pd.to_numeric(df["sir_percent_of_pre"].str.strip('%')) / 100

for name in ["subways_ridership", "buses_ridership", "lirr_ridership", "mn_ridership", 
             "access_a_ride_trips", "bridge_tunnels_traffic", "sir_ridership"]:
    df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
