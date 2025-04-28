import tensorflow as tf
import numpy as np 

def MeanAbsoluteError(predicted_flow, true_flow):
    """
    Calculating mean absolute value
    """
    dif = tf.abs(predicted_flow - true_flow)
    return tf.reduce_mean(dif)

def MeanSquareError(predicted_flow, true_flow):
    """
    Calculating mean absolute value
    """
    dif = tf.math.square(true_flow - predicted_flow)
    return tf.reduce_mean(dif)

def PearsonCorrelationCoefficient(predicted_flow, true_flow): 
    p = predicted_flow - tf.reduce_mean(predicted_flow)
    t = true_flow - tf.reduce_mean(true_flow)

    skibidi = tf.reduce_sum(p * t)
    ohio = tf.math.sqrt(tf.reduce_sum(tf.math.square(p)) * tf.reduce_sum(tf.math.square(t))) 

    return skibidi/ohio
