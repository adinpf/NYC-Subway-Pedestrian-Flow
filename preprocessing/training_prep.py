import pandas as pd
import tensorflow as tf
import networkx as nx
import pickle
import torch

weather_data = pd.read_pickle("data/weather_pandas.pkl")
ridership_data = pd.read_pickle("data/transport_ridership.pkl")
with open("data/subway_network.pkl", "rb") as f:
    graph = pickle.load(f)

# 
# Training turnstile data
turnstile_2023 = pd.read_parquet("data/turnstile_data/2023_turnstile_data.parquet")
weather_2023 = weather_data[weather_data["year"] == 2023]
ridership_2023 = ridership_data[ridership_data["date"].dt.year == 2023]

scalers_2023 = turnstile_2023.groupby('station_complex_id').agg({
    'transfers': ['min', 'max'],
    'ridership': ['min', 'max'],
})
scalers_2023.columns = ['transfers_min', 'transfers_max', 'ridership_min', 'ridership_max']
scalers_2023 = scalers_2023.reset_index()

turnstile_2023 = turnstile_2023.merge(scalers_2023, on='station_complex_id', how='left')
epsilon = 1e-8
turnstile_2023['transfers'] = (turnstile_2023['transfers'] - turnstile_2023['transfers_min']) / (turnstile_2023['transfers_max'] - turnstile_2023['transfers_min'] + epsilon)
turnstile_2023['ridership'] = (turnstile_2023['ridership'] - turnstile_2023['ridership_min']) / (turnstile_2023['ridership_max'] - turnstile_2023['ridership_min'] + epsilon)

ridership_scalers = scalers_2023.set_index('station_complex_id')[['ridership_min', 'ridership_max']].to_dict(orient='index')

# Add min/max to each node
for node_id in graph.nodes:
    scaler = ridership_scalers.get(int(node_id), None)
    graph.nodes[node_id]['ridership_min'] = scaler['ridership_min']
    graph.nodes[node_id]['ridership_max'] = scaler['ridership_max']

turnstile_2023 = turnstile_2023.drop(columns=['transfers_min', 'transfers_max', 'ridership_min', 'ridership_max'])

# Validation/testing turnstile data
turnstile_2024 = pd.read_parquet("data/turnstile_data/2024_turnstile_data.parquet")
weather_2024 = weather_data[weather_data["year"] == 2024]
ridership_2024 = ridership_data[ridership_data["date"].dt.year == 2024]

scalers_2024 = turnstile_2024.groupby('station_complex_id').agg({
    'transfers': ['min', 'max'],
    'ridership': ['min', 'max'],
})
scalers_2024.columns = ['transfers_min', 'transfers_max', 'ridership_min', 'ridership_max']
scalers_2024 = scalers_2024.reset_index()

turnstile_2024 = turnstile_2024.merge(scalers_2024, on='station_complex_id', how='left')
turnstile_2024['transfers'] = (turnstile_2024['transfers'] - turnstile_2024['transfers_min']) / (turnstile_2024['transfers_max'] - turnstile_2024['transfers_min'] + epsilon)
turnstile_2024['ridership'] = (turnstile_2024['ridership'] - turnstile_2024['ridership_min']) / (turnstile_2024['ridership_max'] - turnstile_2024['ridership_min'] + epsilon)

turnstile_2024 = turnstile_2024.drop(columns=['transfers_min', 'transfers_max', 'ridership_min', 'ridership_max'])

print(weather_2024)