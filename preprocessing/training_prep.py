import pandas as pd
import tensorflow as tf

weather_data = tf.tensor(pd.read_csv("data/weather_tensors.pkl")) # Shape is (num_hours, num_features)
with open("data/subway_network.pkl", "rb") as f:
    graph = pickle.load(f)

ridership_data = pd.read_pickle("data/transport_ridership.pkl")

# Training turnstile data
turnstile_2022 = pd.read_pickle("data/turnstile_data/2022_turnstile_data.parquet")
turnstile_2023 = pd.read_pickle("data/turnstile_data/2023_turnstile_data.parquet")

# Testing turnstile data
turnstile_2024 = pd.read_pickle("data/turnstile_data/2024_turnstile_data.parquet")

