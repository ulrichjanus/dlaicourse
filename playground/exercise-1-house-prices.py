# So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, 
# so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

import tensorflow as tf
import numpy as np
from tensorflow import keras

# real house price fct
def hw_function(x):
    return .5 + .5*x

# model
model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
)
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([ 1, 2, 4, 5, 10, 20 ])
ys = hw_function(xs)
model.fit(xs, ys, epochs=500)

# check
x=7
print('x={}, real value y={}, model predicts y={}'.format(x, hw_function(x), model.predict([x])[0][0]))
