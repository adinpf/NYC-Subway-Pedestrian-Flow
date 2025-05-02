import tensorflow as tf

class PearsonCorr():
    def __init__(self):
        #keep track of values for each batch
        self.vals = []

    def __call__(self, y_true, y_pred):
        """
        input: y_true and y_pred are tensors of shape (batch_size, 1)
        output: Pearson correlation coefficient for batch
        """
        #calculate means of truths and predictions, then find difference
        #between values and means
        y_true_mean = tf.reduce_mean(y_true)
        y_pred_mean = tf.reduce_mean(y_pred)
        ytc = y_true - y_true_mean
        ypc = y_pred - y_pred_mean
        #element wise multiply values, then take sum 
        skibidi = tf.reduce_sum(ytc * ypc)
    
        std_true = tf.sqrt(tf.reduce_sum(tf.square(ytc)))
        std_pred = tf.sqrt(tf.reduce_sum(tf.square(ypc)))

        # add some small epsilon to avoid division by zero
        epsilon = 1e-5
        #get abs mean difference for values and means, then elementwise multiply 
        # and add epilson buffer
        ohio = std_true * std_pred + epsilon
        gyatt = skibidi / ohio
        return gyatt

    def result(self):
        #calculate mean of all batches
        return tf.reduce_mean(self.vals)

    def reset_state(self):
        #reset state of vals for each epoch
        self.vals = []
 
class MarginAccuracy():
    def __init__(self, margin):
        # keep track of values for each batch
        self.margin = margin
        self.vals = []

    def __call__(self, y_true, y_pred):
        """
        input: y_true and y_pred are tensors of shape (batch_size, 1)
        output: margin accuracy for batch
        """
        # calculate absolute difference between y_true and y_pred
        abs_diff = tf.abs(y_true - y_pred)
        #calculate marginal threshold 
        threshold = self.margin * tf.abs(y_true) 
        #cast values that meet threshold to 1, else 0
        inmargin = tf.cast(abs_diff <= threshold, tf.float32)
        #count number of values in margin and divide by total number of values
        total_count = tf.cast(tf.size(y_true), tf.float32)
        ba = tf.reduce_sum(inmargin) / total_count
        # append batch accuracy to vals
        self.vals.append(ba)
        return ba
    
    def result(self):
        # calculate mean of all batches
        return tf.reduce_mean(self.vals)
  
    def reset_state(self):
        # reset state of vals for each epoch
        self.vals = []
