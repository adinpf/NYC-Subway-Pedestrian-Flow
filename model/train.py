import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import os
import random
import pickle
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.training_prep import get_turnstile_context, get_external_context

with open("data/subway_network.pkl", "rb") as f:
    graph = pickle.load(f)

def split_external(external_features: pd.DataFrame):
    external_features = external_features.set_index('datetime').copy()
    
    weather_features = external_features[['temp','dew_point','humidity','precipitation','wind_speed','pressure',
                                          'hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6','hour_7',
                                          'hour_8','hour_9','hour_10','hour_11','hour_12','hour_13','hour_14',
                                          'hour_15','hour_16','hour_17','hour_18','hour_19','hour_20','hour_21',
                                          'hour_22','hour_23','month_1','month_2','month_3','month_4','month_5',
                                          'month_6','month_7','month_8','month_9','month_10','month_11','month_12']]
    
    ridership_features = external_features[['0','1','2','3','4','5','6','subways_ridership','subways_percent_of_pre',
                                            'buses_ridership','buses_percent_of_pre','lirr_ridership','lirr_percent_of_pre',
                                            'mn_ridership','mn_percent_of_pre','access_a_ride_trips',
                                            'access_a_ride_percent_of_pre','bridges_tunnels_traffic',
                                            'bridges_tunnels_percent_of_pre','sir_ridership','sir_percent_of_pre']]
    
    ridership_features = ridership_features[ridership_features["subways_percent_of_pre"] != 0]
    
    return ridership_features, weather_features

def make_windows(timestamps, turnstile_df, weather_df):
    # 1) Pivot turnstile into shape [T, N, 2]
    #    T = total timestamps, N = num stations, 2 = [ridership, transfers]
    pivot = turnstile_df.pivot_table(
        index="transit_timestamp",
        columns="station_complex_id",
        values=["ridership","transfers"],
        fill_value=0.0
    )
    # reorder axes to [T, N, 2]
    #   pivot.values is [T, 2, N] so we transpose:
    data_np = pivot.values.transpose(0, 2, 1).astype(np.float32)
    window_len = 24
    # 2) Build sliding windows: shape → [T-window_len+1, window_len, N, 2]
    #    Then we’ll reorder to [T', N, window_len, 2]
    windows = sliding_window_view(data_np, window_shape=window_len, axis=0)
    # windows.shape == (T-window_len+1, window_len, N, 2)
    windows = windows.squeeze()                     # ensure no extra dims
    windows = windows.transpose(0, 2, 1, 3)         # → [T', N, window_len, 2]

    # 3) Pivot weather into [T, W] and then similarly slide
    w_piv = weather_df.set_index("datetime").sort_index().values.astype(np.float32)
    w_windows = sliding_window_view(w_piv, window_shape=window_len, axis=0)
    # w_windows.shape == (T-window_len+1, window_len, W)

    # 4) Build y_true: take ridership at the “present” hour for each station
    #    That’s just data_np[window_len-1 :, :, 0]
    y_np = data_np[window_len-1 :, :, 0]            # [T', N]

    # 5) Zip into TensorFlow-friendly tuples
    T_prime = windows.shape[0]
    out = []
    for i in range(T_prime):
        ts = pivot.index[i + window_len]            # corresponding timestamp
        temporal = tf.convert_to_tensor(windows[i], dtype=tf.float32)     # [N,24,2]
        weather  = tf.convert_to_tensor(w_windows[i], dtype=tf.float32)   # [24,W]
        y_true   = tf.convert_to_tensor(y_np[i], dtype=tf.float32)        # [N]
        out.append((ts, temporal, weather, y_true))

    return out
    # weather_list = []
    # temporal_list = []
    # y_true_list = []
    
    # for ts in tqdm(timestamps, desc="Making windows"):
    #     if (ts.day == 2 and ts.month == 1):
    #         print("did 1")
    #     ts_list = []
    #     y_true_per_ts = []  # collect all y_true for this timestamp
        
    #     weather_list.append(
    #         tf.convert_to_tensor(get_external_context(ts, weather_features, 24), 
    #                              dtype=tf.float32)
    #     )
        
    #     for node_id in graph.nodes:
    #         tens = tf.convert_to_tensor(get_turnstile_context(ts, int(node_id), temporal_features, 24),
    #                                             dtype=tf.float32)
    #         if len(tens) != 24:
    #             print(tf.shape(tens))
    #         ts_list.append(tens)
            
    #         match = temporal_features[
    #             (temporal_features["transit_timestamp"] == ts) & 
    #             (temporal_features["station_complex_id"] == int(node_id))
    #         ]
    #         if not match.empty:
    #             y_true_value = match["ridership"].values[0] 
    #         else:
    #             y_true_value = 0.0  
            
    #         y_true_per_ts.append(y_true_value)
        
    #     temporal_list.append(tf.convert_to_tensor(ts_list, dtype=tf.float32)) 
    #     y_true_list.append(tf.convert_to_tensor(y_true_per_ts, dtype=tf.float32))
    
    # return list(zip(timestamps, temporal_list, weather_list, y_true_list))



def train(model, epochs, batch_size, data):
    # spatial_features, temporal_features, external_features, weather_features, A = model_inputs
    
    spatial_features, temporal_features, external_features, A = data
    ridership_features, weather_features = split_external(external_features)
    spatial_features = tf.convert_to_tensor(spatial_features, dtype=tf.float32)
    
    timestamps = weather_features.index
    timestamps = timestamps[(timestamps.month > 1) | (timestamps.day > 1)]
    print("\nmaking window")
    windows = make_windows(timestamps, temporal_features, weather_features)
    
    for epoch in range(epochs):
        # shuffle timestamps each epoch
        random.shuffle(windows)
            
        # "batching"
        for i in tqdm(range(0, len(windows), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = windows[i:i + batch_size]

            with tf.GradientTape() as tape:
                total_loss = 0.0
                for (ts, temporal_context, weather_context, y_true) in batch:
                    mask = (ridership_features.index.year == ts.year) & (ridership_features.index.day == ts.day) & (ridership_features.index.month == ts.month)
                    
                    arr = ridership_features.loc[mask].values
                    ridership_vector = tf.convert_to_tensor(arr, dtype=tf.float32)
                    ridership_vector = tf.expand_dims(ridership_vector, axis=0)
                    weather_context = tf.expand_dims(weather_context, axis=0)
                    y_true = tf.expand_dims(y_true, axis=-1)
                    
                    # forward pass on one window
                    y_pred = model(spatial_features, temporal_context, ridership_vector, weather_context, A, training=True)  # [N,1]
                    total_loss += model.loss(y_true, y_pred)

                # average the loss over the K windows
                total_loss /= tf.cast(len(batch), tf.float32)

            # backprop once
            grads = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    