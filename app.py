import struct
from flask import Flask, request, json

# Image handler lib 
from PIL import Image
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# input image dimensions 
IMG_ROWS, IMG_COLS = 32, 32

DIC_POKEMON_TYPE = {
    'Bug' : 0,
    'Dark' : 1,
    'Dragon' : 2,
    'Electric' : 3,
    'Fairy' : 4,
    'Fighting' : 5,
    'Fire' : 6,
    'Flying' : 7,
    'Ghost' : 8,
    'Grass' : 9,
    'Ground' : 10,
    'Ice' : 11,
    'Normal' : 12,
    'Poison' : 13,
    'Psychic' : 14,
    'Rock' : 15,
    'Steel' : 16,
    'Water' : 17,
    18 : 18
}

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

@app.route('/perceptron', methods=["POST"])
def analyse_image_perceptron():
    file = request.files['image']

    if not file:
        return app.response_class(
            response="Empty image", 
            status=400, 
            mimetype='application/json'
        )

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
    
    label_predict = DIC_POKEMON_TYPE[np.argmax(labels_predict)]

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

    MULTI_LAYER_MODEL = keras.models.load_model('models/pokemon_multi_layer.h5')
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

    label_predict = DIC_POKEMON_TYPE[np.argmax(labels_predict)]
    
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

    CNN_MODEL = keras.models.load_model('models/pokemon_cnn.h5')
    CNN_MODEL._make_predict_function()

    try:
        file_image_rgb = convert_io_string_to_image_rgb(file.stream)
    
    except BadDimensionError as error:
        return app.response_class(
            response=json.dumps(error.message), 
            status=400, 
            mimetype='application/json'
        )

    labels_predict = CNN_MODEL.predict(file_image_rgb)

    label_predict = DIC_POKEMON_TYPE[np.argmax(labels_predict)]

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

