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

if __name__ == "__main__":
    spatial_data = pd.read_parquet("data/final_data/spatial_data.parquet")
    external_2023 = pd.read_parquet("data/final_data/external_2023.parquet")
    external_2024 = pd.read_parquet("data/final_data/external_2024.parquet")
    temporal_2023 = pd.read_parquet("data/final_data/temporal_2023.parquet")
    temporal_2024 = pd.read_parquet("data/final_data/temporal_2024.parquet")
    graph = pd.read_pickle("data/subway_network.pkl")
    
    model = DSTGCN((len(spatial_data.columns),
                    len(temporal_2023.columns),
                    21, 
                    42,
                    len(spatial_data)))
    
    adjacency_matrix = build_adjacency_matrix(graph)
    
    train(model=model, epochs=50, batch_size=30, data=(spatial_data, temporal_2023, external_2023, adjacency_matrix))
    
    
