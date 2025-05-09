import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from architecture.DSTGCN import DSTGCN  
from train import train
from preprocessing.build_adjacency_matrix import build_adjacency_matrix
from test import test
import time
from architecture.metrics import PearsonCorr, MarginAccuracy
from make_windows import make_windows, split_external

if __name__ == "__main__":
    start_proc = time.time()
    spatial_data = pd.read_parquet("data/final_data/spatial_data.parquet")
    external_2023 = pd.read_parquet("data/final_data/external_2023.parquet")
    external_2024 = pd.read_parquet("data/final_data/external_2024.parquet")
    temporal_2023 = pd.read_parquet("data/final_data/temporal_2023.parquet")
    temporal_2024 = pd.read_parquet("data/final_data/temporal_2024.parquet")
    graph = pd.read_pickle("data/subway_network.pkl")
    
    windows_2023 = make_windows(temporal_2023, external_2023, graph)
    windows_2024 = make_windows(temporal_2024, external_2024, graph)
    
    print(f'loaded files in {time.time()-start_proc:4f}s')
    ridership_2023, weather_2023 = split_external(external_2023)
    model_load = time.time()
    model = DSTGCN((len(temporal_2023.columns),
                    len(ridership_2023.columns), 
                    len(weather_2023.columns),
                    1))
    print(f'instantiated model in {time.time() - model_load:4f}s')
    
    A = build_adjacency_matrix(graph)
    epochs = 10

    batch_size = 128
    # exit
    print('starting to train')
    training_start = time.time()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsoluteError()
        )
    
    average_training_loss, average_training_pearson, average_training_accuracy = train(model=model, 
          epochs=epochs, 
          batch_size=batch_size, 
          data=(spatial_data, windows_2023, A))
    # model.save_weights('model/dstgcn_full_model_7epochs')
    # model.load_weights('model/dstgcn_full_model_5epochs')

    print(f'finished training in {time.time()-training_start:4f}s')
    print('starting to test')
    average_batch_loss, average_pearson, average_accuracy = test(model=model, 
                              batch_size=batch_size, 
                              data=(spatial_data, windows_2024, A))
    print(f"The average testing loss is {average_batch_loss}, the average pearson correlation is {average_pearson}, and the average accuracy is {average_accuracy}")
    
    
