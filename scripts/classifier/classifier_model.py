import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sys
import os
print(sys.path)
print(os.getcwd())
sys.path.insert(0, os.getcwd())

def classifier(
        compile=True,
        input_shape= tuple(),
        dense_layer_size = 50,
        layers=1,
        activation="relu",
        num_classes=1,
        output_name= 'output'
):
    # create the model
    model = Sequential()
    model.add(Dense(dense_layer_size, input_shape=input_shape, activation=activation))
    for i in range(layers-1):
        model.add(Dense(dense_layer_size, activation=activation))

    model.add(Dense(num_classes, activation='softmax', name=output_name))

    if compile:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

    return model

# ## Old simple model
# def classifier_model(
#         compile=True,
#         input_shape=tuple(),
#         nb_phonemes =1,
#         output_name="output"
# ):
#     # create the model
#     model = Sequential()
#     # model.add(Dense(30, input_shape=input_shape, activation='relu'))
#     model.add(Dense(nb_phonemes, input_shape=input_shape, activation='softmax', name=output_name))
#
#     if compile:
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         model.summary()
#
#     return model

