# Course 1 - part 2 - Lessen 2
# The Hello World of Deep Learning with Neural Networks¶

import tensorflow as tf
import numpy as np
from tensorflow import keras


# real function 
def hw_function(x):
    return (2 * x) - 1

# the neural network 
model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
)
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=500)

# compare prediction to real value
x=10
print('x={}, real value y={}, model predicts y={}'.format(x, hw_function(x), model.predict([x])[0][0]))
