import tensorflow as tf
import tensorflow.keras.backend as K

def custom_mse(y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_diff = y_true - y_pred
    
    tf_ones = tf.ones(tf.shape(y_diff)[0], 1)
    
    w = tf.stack([(y_true[:, 1] + 1)/2, tf_ones])
    w = tf.transpose(w)
    
    y_diff_w = tf.math.multiply(y_diff, w)
    
    return K.mean(K.square(y_diff_w))