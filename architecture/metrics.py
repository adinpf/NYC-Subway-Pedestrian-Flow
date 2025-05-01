import tensorflow as tf
import numpy as np

class PearsonCorr():
    def __init__(self):
        self.vals = []

    def __call__(self, y_true, y_pred):

        y_true_mean = tf.reduce_mean(y_true)
        y_pred_mean = tf.reduce_mean(y_pred)
        ytc = y_true - y_true_mean
        ypc = y_pred - y_pred_mean
        skibidi = tf.reduce_sum(ytc * ypc)
    
        std_true = tf.sqrt(tf.reduce_sum(tf.square(ytc)))
        std_pred = tf.sqrt(tf.reduce_sum(tf.square(ypc)))

        ohio = std_true * std_pred
        gyatt = skibidi / ohio
        
        self.vals.append(gyatt)

        return gyatt

    def result(self):
        return tf.reduce_mean(self.vals)

    
    def reset_state(self):
        self.vals = []
 
class MarginAccuracy():
    def __init__(self, margin):
      
        self.margin = margin
        self.vals = []

    def __call__(self, y_true, y_pred):

        abs_diff = tf.abs(y_true - y_pred)
        threshold = self.margin * tf.abs(y_true) 
        inmargin = tf.cast(abs_diff <= threshold, tf.float32)
        total_count = tf.cast(tf.size(y_true), tf.float32)
        ba = tf.reduce_sum(inmargin) / total_count
        self.vals.append(ba)
        return ba
    
    def result(self):
        return tf.reduce_mean(self.vals)
  
    def reset_state(self):
        self.vals = []


"""
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="mse", 
    metrics=[PearsonCorr(), MarginAccuracy(margin=0.05)]  # 5% margin
)
"""

