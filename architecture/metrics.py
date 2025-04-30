import tensorflow as tf
import numpy as np


class PearsonCorr(tf.keras.metrics.Metric):
    def __init__(self, name='pearson_correlation', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_over_batches = self.add_weight(name="total", initializer="zeros")
        self.num_batches = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred):
        if len(tf.shape(y_true)) > 1:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
        
        y_true_mean = tf.reduce_mean(y_true)
        y_pred_mean = tf.reduce_mean(y_pred)
        
        ytc = y_true - y_true_mean
        ypc = y_pred - y_pred_mean
        
        skibidi = tf.reduce_sum(ytc * ypc)
        
        epsilon = 1e-3
        std_true = tf.sqrt(tf.reduce_sum(tf.square(ytc)) + epsilon)
        std_pred = tf.sqrt(tf.reduce_sum(tf.square(ypc)) + epsilon)
        ohio = std_true * std_pred
        
        gyatt = skibidi / ohio
        
        self.total_over_batches.assign_add(gyatt)
        self.num_batches.assign_add(1.0)
        
        return gyatt

    def result(self):
        return self.total_over_batches / self.num_batches
    
    def reset_state(self):
        self.total_over_batches.assign(0.0)
        self.num_batches.assign(0.0)


class MarginAccuracy(tf.keras.metrics.Metric):
    def __init__(self, margin=0.05, name='margin_accuracy', **kwargs):
      
        super().__init__(name=name, **kwargs)
        self.margin = margin
        self.num_correct = self.add_weight(name="num_correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        if len(tf.shape(y_true)) > 1:
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])

        abs_diff = tf.abs(y_true - y_pred)
        
        threshold = self.margin * tf.abs(y_true)
        
        inmargin = tf.cast(abs_diff <= threshold, tf.float32)
        
  
        if sample_weight is not None:
            inmargin = inmargin * sample_weight
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            total_count = tf.cast(tf.size(y_true), tf.float32)
            self.total.assign_add(total_count)
        
        self.num_correct.assign_add(tf.reduce_sum(inmargin))
        ba = tf.reduce_sum(inmargin) / tf.cast(tf.size(y_true), tf.float32)
        return ba
    def result(self):
        return self.num_correct / self.total
    
    def reset_state(self):
        self.num_correct.assign(0.0)
        self.total.assign(0.0)

"""
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="mse", 
    metrics=[PearsonCorr(), MarginAccuracy(margin=0.05)]  # 5% margin
)
"""