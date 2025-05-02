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
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from architecture.metrics import PearsonCorr, MarginAccuracy

def train(model, epochs, batch_size, data):
    # spatial_features, temporal_features, external_features, weather_features, A = model_inputs
    spatial_features, windows, A = data
    
    spatial_features = tf.convert_to_tensor(spatial_features, dtype=tf.float32)
    
    loss_l = []
    pearson_l = []
    margin_l = []

    pearson = PearsonCorr()
    margin = MarginAccuracy(margin=5.0)

    
    for epoch in range(epochs):
        # shuffle timestamps each epoch
        random.shuffle(windows)
        epoch_loss = 0
        batch_count = 0
        # "batching"
        for i in tqdm(range(0, len(windows), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = windows[i:i + batch_size]

            with tf.GradientTape() as tape:
                total_loss = 0.0
                for (ts, temporal_context, weather_context, ridership_vector, y_true) in batch:
                    # forward pass on one window
                    y_pred = model((spatial_features, temporal_context, ridership_vector, weather_context, A), training=True)  # [N,1]
                    # print(y_pred[tf.math.is_nan(y_pred)])
                    total_loss += model.loss(y_true, y_pred)
                    pearson(y_true, y_pred)  
                    margin(y_true, y_pred) 

                # average the loss over the K windows
                total_loss /= tf.cast(len(batch), tf.float32)
            
            batch_count += 1
            epoch_loss += total_loss
            # backprop once
            grads = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        average_epoch_loss = epoch_loss / batch_count
        loss_l.append(average_epoch_loss)
        pearson_l.append(pearson.result())
        margin_l.append(margin.result())
        print(f"Average loss for epoch {epoch+1}: {average_epoch_loss}, average pearson: {pearson.result()}, average margin: {margin.result()}\n")

        pearson.reset_state()
        margin.reset_state()
    

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(loss_l)
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.subplot(1, 3, 2)
    plt.plot(pearson_l)
    plt.title('Pearson Correlation Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('PCC')
    plt.subplot(1, 3, 3)
    plt.plot(margin_l)
    plt.title('Marginal Accuracy (5%)')
    plt.xlabel('Epoch')
    plt.ylabel('Marginal Accuracy')
    plt.tight_layout()
    plt.savefig('metrics/metrics_train.png')
    plt.close()
    print(f"Average loss for training: {tf.reduce_mean(loss_l)}, average pearson: {tf.reduce_mean(pearson_l)}, average margin: {tf.reduce_mean(margin_l)}\n")
    return tf.reduce_mean(loss_l), tf.reduce_mean(pearson_l), tf.reduce_mean(margin_l)
    
    
    