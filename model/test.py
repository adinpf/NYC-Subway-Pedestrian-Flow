import tensorflow as tf
from tqdm import tqdm
import pickle
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import random
from architecture.metrics import PearsonCorr, MarginAccuracy
import matplotlib.pyplot as plt

with open("data/subway_network.pkl", "rb") as f:
    graph = pickle.load(f)

def test(model, batch_size, data):
    # spatial_features, temporal_features, external_features, weather_features, A = model_inputs
    spatial_features, windows, A = data
    
    spatial_features = tf.convert_to_tensor(spatial_features, dtype=tf.float32)
    pearson = PearsonCorr()
    margin = MarginAccuracy(margin=5.0)
    
    loss_l = []
    
    total_loss = 0.0
    total_samples = 0.0
    # "batching"
    for i in tqdm(range(0, len(windows), batch_size)):
        batch = windows[i:i + batch_size]

        batch_loss = 0.0
        for (ts, temporal_context, weather_context, ridership_vector, y_true) in batch:
            # forward pass on one window
            y_pred = model((spatial_features, temporal_context, ridership_vector, weather_context, A), training=False)  # [N,1]
            loss = model.loss(y_true, y_pred)
            pearson(y_true, y_pred)
            margin(y_true, y_pred)
            loss_l.append(loss)
            
            total_loss += loss
            total_samples += 1
    
    average_loss = total_loss / total_samples
    average_pearson = pearson.result()
    average_margin = margin.result()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(loss_l)
    plt.title('Mean Absolute Error')
    plt.xlabel('Batch')
    plt.ylabel('MAE')
    plt.subplot(1, 3, 2)
    plt.plot(pearson.vals)
    plt.title('Pearson Correlation Coefficient')
    plt.xlabel('Batch')
    plt.ylabel('PCC')
    plt.subplot(1, 3, 3)
    plt.plot(margin.vals)
    plt.title('Marginal Accuracy (5%)')
    plt.xlabel('Batch')
    plt.ylabel('Marginal Accuracy')
    plt.tight_layout()
    plt.savefig('metrics/metrics_test.png')
    plt.close()
    return average_loss, average_pearson, average_margin
