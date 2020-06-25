import struct
from flask import Flask, request, json

# Image handler lib 
from PIL import Image
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

app = Flask(__name__)

# input image dimensions 
IMG_ROWS, IMG_COLS = 32, 32

ARRAY_POKEMON_TYPE = [
    'Bug',
    'Dark',
    'Dragon',
    'Electric',
    'Fairy',
    'Fighting',
    'Fire',
    'Flying',
    'Ghost',
    'Grass',
    'Ground',
    'Ice',
    'Normal',
    'Poison',
    'Psychic',
    'Rock',
    'Steel',
    'Water',
    18
]

class BadDimensionError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "Bab dimension error for given image"
        
    def __str__(self):
        if self.message:
            return 'BadDimensionError, {0} '.format(self.message)
        else:
            return 'BadDimensionError has been raised'


def convert_io_string_to_image_rgb(io_string):
    file = Image.open(io_string)

    if file.size != (IMG_ROWS, IMG_COLS):
        raise BadDimensionError("The image should have a size of: " + str(IMG_ROWS) + "X" + str(IMG_COLS))

    # Virer alpha
    file = file.convert('RGB')
    np_array_file = np.array( file, dtype='uint8' )

    # Flatten
    np_array_file = np_array_file.reshape((1, -1))
    
    # Divide by pixel
    np_array_file = np_array_file / 255

    return np_array_file

def convert_io_string_to_image_rgb_cnn(io_string):
    file = Image.open(io_string)

    if file.size != (IMG_ROWS, IMG_COLS):
        raise BadDimensionError("The image should have a size of: " + str(IMG_ROWS) + "X" + str(IMG_COLS))

    # Virer alpha
    file = file.convert('RGB')
    data = np.array( file, dtype='uint8' )

    if K.image_data_format() == 'channels_first':
        data = data.reshape(1, 3, IMG_ROWS, IMG_COLS)
    else:
        data = data.reshape(1, IMG_ROWS, IMG_COLS, 3)
    
    data = data.astype('float32')
    data /= 255

    return data


@app.route('/perceptron', methods=["POST"])
def analyse_image_perceptron():
    file = request.files['image']

    if not file:
        return app.response_class(
            response="Empty image", 
            status=400, 
            mimetype='application/json'
        )

    tf.keras.backend.clear_session()
    SINGLE_LAYER_MODEL = tf.keras.models.load_model('models/pokemon_linear_perceptron_model.h5')
    SINGLE_LAYER_MODEL._make_predict_function()

    try:
        file_image_rgb = convert_io_string_to_image_rgb(file.stream)
    
    except BadDimensionError as error:
        return app.response_class(
            response=json.dumps(error.message), 
            status=400, 
            mimetype='application/json'
        )

    labels_predict = SINGLE_LAYER_MODEL.predict(file_image_rgb)
    
    label_predict = ARRAY_POKEMON_TYPE[np.argmax(labels_predict)]

    if label_predict == 18:
        return app.response_class(
            response=json.dumps("Well ... sorry i don't found"),
            status=200,
            mimetype='application/json'
        )

    return app.response_class(
            response=json.dumps(label_predict),
            status=200,
            mimetype='application/json'
        )

@app.route('/multi-layers-perceptron', methods=["POST"])
def analyse_image_multi_layers_perceptron():
    file = request.files['image']

    if not file:
        return app.response_class(
            response="Empty image", 
            status=400, 
            mimetype='application/json'
        )

    tf.keras.backend.clear_session()
    MULTI_LAYER_MODEL = tf.keras.models.load_model('models/pokemon_multi_layer.h5')
    MULTI_LAYER_MODEL._make_predict_function()

    try:
        file_image_rgb = convert_io_string_to_image_rgb(file.stream)
    
    except BadDimensionError as error:
        return app.response_class(
            response=json.dumps(error.message), 
            status=400, 
            mimetype='application/json'
        )

    labels_predict = MULTI_LAYER_MODEL.predict(file_image_rgb)


    label_predict = ARRAY_POKEMON_TYPE[np.argmax(labels_predict)]
    
    if label_predict == 18:
        return app.response_class(
            response=json.dumps("Well ... sorry i don't found"),
            status=200,
            mimetype='application/json'
        )

    return app.response_class(
            response=json.dumps(label_predict),
            status=200,
            mimetype='application/json'
        )

@app.route('/cnn', methods=["POST"])
def analyse_image_cnn():
    file = request.files['image']

    if not file:
        return app.response_class(
            response="Empty image", 
            status=400, 
            mimetype='application/json'
        )

    tf.keras.backend.clear_session()
    CNN_MODEL = tf.keras.models.load_model('models/pokemon_cnn.h5')
    CNN_MODEL._make_predict_function()

    try:
        file_image_rgb = convert_io_string_to_image_rgb_cnn(file.stream)
    
    except BadDimensionError as error:
        return app.response_class(
            response=json.dumps(error.message), 
            status=400, 
            mimetype='application/json'
        )

    labels_predict = CNN_MODEL.predict(file_image_rgb)

    label_predict = ARRAY_POKEMON_TYPE[np.argmax(labels_predict)]

    if label_predict == 18:
        return app.response_class(
            response=json.dumps("Well ... sorry i don't found"),
            status=200,
            mimetype='application/json'
        )

    return app.response_class(
            response=json.dumps(label_predict),
            status=200,
            mimetype='application/json'
        )
