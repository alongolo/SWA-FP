""" Tensorflow Keras SWA Object
"""

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from swa_callback import create_swa_improved_callback_class

SWA_improved = create_swa_improved_callback_class(K, Callback, BatchNormalization)
