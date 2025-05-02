import tensorflow as tf
from tqdm import tqdm
import pickle
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import random

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
    
    ridership_features = ridership_features[ridership_features.index.hour == 23]
    
    return ridership_features, weather_features

def make_windows(temporal_features, external_features, graph):
    ridership_features, weather_features = split_external(external_features)
    
    timestamps = weather_features.index
    timestamps = np.array(timestamps[(timestamps.month > 1) | (timestamps.day > 1)])
    indices_mask = np.random.permutation(len(timestamps))
    timestamps = timestamps[indices_mask]
    
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
    
    windows = []
    # timestamps = timestamps[:1000]
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
        
        weather_context = tf.convert_to_tensor(weather_np, dtype=tf.float32)  # [24, F_ext]
        weather_context = tf.expand_dims(weather_context, axis=0)            # [1, 24, F_ext]
        weather_context = tf.transpose(weather_context, perm=[0, 2, 1])      # [1, F_ext, 24]
        weather_context = tf.where(tf.math.is_nan(weather_context), tf.zeros_like(weather_context), weather_context)

        y_true = tf.convert_to_tensor(y_true_np, dtype=tf.float32)           # [N]
        y_true = tf.expand_dims(y_true, axis=-1)   
        
        
        windows.append((
            ts,
            tf.convert_to_tensor(temp_np,    dtype=tf.float32),  # [N,24,2]
            tf.convert_to_tensor(weather_np, dtype=tf.float32),  # [24,F_ext]
            ridership_vector,                                    # [F_rid, 1]
            tf.convert_to_tensor(y_true_np,  dtype=tf.float32)   # [N]
        ))
        
    return windows