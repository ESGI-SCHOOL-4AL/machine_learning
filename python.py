import struct

# Image handler lib 
from PIL import Image
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow import keras

# input image dimensions 
IMG_ROWS, IMG_COLS = 32, 32

SINGLE_LAYER_MODEL = tf.keras.models.load_model('models/pokemon_linear_perceptron_model.h5')

def convert_io_string_to_image_rgb(io_string):
    file = Image.open(io_string)

    if file.size != (IMG_ROWS, IMG_COLS):
        raise Exception("The image should have a size of: " + str(IMG_ROWS) + "X" + str(IMG_COLS))

    # Virer alpha
    file = file.convert('RGB')
    np_array_file = np.array( file, dtype='uint8' )

    # Flatten
    np_array_file = np_array_file.reshape((1, -1))
    
    # Divide by pixel
    np_array_file = np_array_file / 255

    return np_array_file


file_image_rgb = convert_io_string_to_image_rgb('test_sample/005.png')
label_predict = SINGLE_LAYER_MODEL.predict(file_image_rgb)
    
print(label_predict)