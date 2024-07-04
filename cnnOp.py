import tensorflow as tf 
import keras 

class Conv2DPadded(keras.layers.Layer):
    def __init__(self, filter, kernel, stride, paddingAmt, activation=keras.layers.LeakyReLU(), padType='SYMMETRIC'):
        super().__init__()
        self.filter = filter 
        self.kernel = kernel 
        self.stride = stride 
        self.paddingAmt = paddingAmt
        self.padType = padType
        self.activation = activation
        self.conv2d = keras.layers.Conv2D(filter, kernel, stride, "valid", activation=activation)

    def call(self, x):
        x = tf.pad(x, ((0,0), (self.paddingAmt, self.paddingAmt), (self.paddingAmt, self.paddingAmt), (0,0)), self.padType)
        x = self.conv2d(x)
        return x 
    
    def get_config(self):
        return {
            "filter": self.filter,
            "kernel": self.kernel,
            "stride": self.stride,
            "paddingAmt": self.paddingAmt,
            "activation": self.activation,
            "padType": self.padType
        }

class Conv3DPadded(keras.layers.Layer):
    def __init__(self, filters, kernel, strides, paddingAmt, activation=keras.layers.LeakyReLU(), padType='SYMMETRIC'):
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.paddingAmt = paddingAmt
        self.padType = padType
        self.activation = activation
        self.conv3d = keras.layers.Conv3D(filters, kernel, strides, "valid", activation=activation)

    def call(self, x):
        pad_width = ((0, 0),
                     (self.paddingAmt, self.paddingAmt),
                     (self.paddingAmt, self.paddingAmt),
                     (self.paddingAmt, self.paddingAmt),
                     (0, 0))
        x = tf.pad(x, pad_width, self.padType)
        x = self.conv3d(x)
        return x
