import tensorflow as tf
from train import split_external
from tqdm import tqdm
import pickle
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import random

with open("data/subway_network.pkl", "rb") as f:
    graph = pickle.load(f)

temporal_features = pd.read_parquet("data/final_data/temporal_2024.parquet")
external_features = pd.read_parquet("data/final_data/external_2024.parquet")
ridership_features, weather_features = split_external(external_features)

tf_df = temporal_features.copy()
if not isinstance(tf_df.index, pd.DatetimeIndex):
    tf_df.set_index('transit_timestamp', inplace=True)
tf_df.sort_index(inplace=True)

station_ids = [int(sid) for sid in graph.nodes]

station_windows = {}
all_ts = set()
for sid, grp in tf_df.groupby('station_complex_id'):
    arr = grp[['ridership','transfers']].values       
    wins = sliding_window_view(arr, window_shape=24, axis=0)  
    ts_ends = grp.index[(grp.index.month > 1) | (grp.index.day > 1)]                             
    station_windows[sid] = dict(zip(ts_ends, wins))
    all_ts.update(ts_ends)
    
all_ts = sorted(all_ts)

for sid in station_ids:
    if sid not in station_windows:
        station_windows[sid] = {}
    for ts in all_ts:
        if ts not in station_windows[sid]:
            station_windows[sid][ts] = np.zeros((24, 2))

weather_arr = weather_features.values                   
ext_wins   = sliding_window_view(weather_arr, 24, axis=0)  
ext_ts     = weather_features.index[(weather_features.index.month > 1) | (weather_features.index.day > 1)]
external_windows = dict(zip(ext_ts, ext_wins))

F_ext = ext_wins.shape[2] if ext_wins.ndim == 3 else ext_wins.shape[1]
for ts in all_ts:
    if ts not in external_windows:
        external_windows[ts] = np.zeros((24, F_ext))

y_true_df = (
    tf_df
    .reset_index()
    .pivot(
        index='transit_timestamp',
        columns='station_complex_id',
        values='ridership'
    )
    .reindex(columns=station_ids, fill_value=0)
)

def make_windows(timestamps):
    windows = []
    timestamps = timestamps[:100]
    for ts in tqdm(timestamps, desc="Making windows"):
        ts = pd.Timestamp(ts)
        temp_np    = np.stack([station_windows[sid][ts] for sid in station_ids])  # (N,24,2)
        weather_np = external_windows[ts]                                         # (24,F_ext)
        y_true_np  = y_true_df.loc[ts, station_ids].values.astype(np.float32)     # (N,)
        
        day_before = ts - pd.Timedelta(days=1)

        mask = (ridership_features.index.year == day_before.year) & \
        (ridership_features.index.month == day_before.month) & \
        (ridership_features.index.day == day_before.day)
        
        arr = ridership_features.loc[mask].values
        ridership_vector = tf.convert_to_tensor(arr, dtype=tf.float32)
        
        windows.append((
            ts,
            tf.convert_to_tensor(temp_np,    dtype=tf.float32),  # [N,24,2]
            tf.convert_to_tensor(weather_np, dtype=tf.float32),  # [24,F_ext]
            ridership_vector,                                    # [F_rid, 1]
            tf.convert_to_tensor(y_true_np,  dtype=tf.float32)   # [N]
        ))
    return windows

def test(model, batch_size, data):
    # spatial_features, temporal_features, external_features, weather_features, A = model_inputs
    
    spatial_features, temporal_features, ridership_features, weather_features, A = data
    spatial_features = tf.convert_to_tensor(spatial_features, dtype=tf.float32)
    
    timestamps = weather_features.index
    timestamps = np.array(timestamps[(timestamps.month > 1) | (timestamps.day > 1)])
    indices_mask = np.random.permutation(len(timestamps))
    timestamps = timestamps[indices_mask]
    windows = make_windows(timestamps)
    
    total_loss = 0.0
    total_samples = 0.0
    # "batching"
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]

        batch_loss = 0.0
        for (ts, temporal_context, weather_context, ridership_vector, y_true) in batch:
            weather_context = tf.expand_dims(weather_context, axis=0)
            weather_context = tf.transpose(weather_context, perm=[0, 2, 1])
            y_true = tf.expand_dims(y_true, axis=-1)
            # forward pass on one window
            y_pred = model((spatial_features, temporal_context, ridership_vector, weather_context, A), training=False)  # [N,1]
            
            loss = model.loss(y_true, y_pred)
            if not(tf.math.is_nan(loss)):
                total_loss += loss
                total_samples += 1
            # else:
            #     print(f"y_pred: {y_pred}")
            #     print(f"timestamp: {ts}")
            #     print(f"temporal_context shape: {temporal_context.shape}")  # Expect (N, 24, 2)
            #     print(f"ridership_vector shape: {ridership_vector.shape}")  # Expect (F_ridership,)
            #     print(f"weather_context shape: {weather_context.shape}")    # Expect (1, 24, F_ext)
            #     print(f"spatial_features shape: {spatial_features.shape}")  # Expect (N, F_spatial)
            # print(f"Y_pred is {y_pred}")
            # print(f"Y_true is {y_true}")

        # average the loss over the K windows
        # batch_loss /= tf.cast(len(batch), tf.float32)
        # batch_losses.append(batch_loss)
    
    average_loss = total_loss / total_samples
    print(f"Test loss: {average_loss}")
    return average_loss
