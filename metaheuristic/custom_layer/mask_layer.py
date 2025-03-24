import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable          # type: ignore
from tensorflow.keras.layers import Layer, Flatten, Dense, Input        # type: ignore
from tensorflow.keras.models import load_model, Model                   # type: ignore

@register_keras_serializable(package="Custom", name="MaskLayer")
class MaskLayer(Layer):

    def __init__(self, mask=None, name="mask", **kwargs):
        super(MaskLayer, self).__init__(name=name, **kwargs)
        self.mask = np.array(mask, dtype=np.int32)
        self.feature_dimension = tf.reduce_sum(self.mask).numpy()
    
    def build(self, input_shape):
        self.mask_variable = self.add_weight(
            shape=self.mask.shape,
            initializer=tf.constant_initializer(self.mask),
            trainable=False,
            name="mask_variable",
        )

    def call(self, inputs):
        
        mask_boolean = tf.cast(self.mask_variable, tf.bool)
        mask_boolean = tf.reshape(mask_boolean, [1, -1])
        mask_boolean = tf.tile(mask_boolean, [tf.shape(inputs)[0], 1]) 
        masked_input = tf.boolean_mask(inputs, mask_boolean)

        batch_size = tf.shape(inputs)[0]
        feature_count = tf.reduce_sum(tf.cast(mask_boolean[0], tf.int32))
        return tf.reshape(masked_input, [batch_size, feature_count])
        
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.feature_dimension)  
    
    def set_mask(self, new_mask):
        self.mask_variable.assign(new_mask)

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        config.update({"mask": self.mask.tolist()})
        return config
    
    def get_weights(self):
        return [self.mask_variable.numpy()]

    def set_weights(self, weights):
        self.mask_variable.assign(weights[0])
        
    @classmethod
    def from_config(cls, config):
        mask = tf.convert_to_tensor(config['mask'], dtype=tf.float32)
        return cls(mask)
