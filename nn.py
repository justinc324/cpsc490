#!/usr/bin/python3

import keras
import numpy
from keras.callbacks import ModelCheckpoint
from keras import Sequential
from keras.utils import multi_gpu_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional


class MusicRNN:

    def __init__(self, notes_manager, epochs=100, batch_size=32):

        # notes manager
        self.notes_manager = notes_manager
        self.num_features = notes_manager.num_features

        # model architecture
        # self._model = Sequential()
        # self._model.add(Bidirectional(LSTM(512, return_sequences=True, activation='relu', input_shape=(16, self.num_features))))
        # self._model.add(Dropout(.2))
        # self._model.add(Bidirectional(LSTM(512)))
        # self._model.add(Dense(self.num_features, activation='softmax'))
        # self._model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        self._model = Sequential()
        self._model.add(Bidirectional(LSTM(256, return_sequences=True, activation='relu', input_shape=(16, self.num_features))))
        self._model.add(Dropout(.2))
        self._model.add(Bidirectional(LSTM(256)))
        self._model.add(Dense(self.num_features, activation='sigmoid'))

        try:
            self._model = multi_gpu_model(self._model)
        except:
            pass

        self._model.compile(loss='categorical_crossentropy', optimizer='adam')

        self._epochs = epochs
        self._batch_size = batch_size

    def train(self, training_input, training_output, filename=None):
        """
        Trains the RNN with the given training data
        """
        # if a file name is given, save the weights during training
        if filename:
            filepath = "{}.weights.best.hdf5".format(filename)
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            self._model.fit(training_input, training_output, callbacks=callbacks_list, validation_split=.33,
                            epochs=self._epochs, batch_size=self._batch_size, verbose=0)
        else:
            self._model.fit(training_input, training_output, epochs=self._epochs,
                            validation_split=.33, batch_size=self._batch_size)

    def predict_note(self, prediction_input):
        """
        Predicts a single note (pitch, len) based on input to the trained
        NN
        """
        encoder = self.notes_manager.one_hot_encoder
        classes = encoder.classes_  # array of classes of pitches and lengths
        num_lengths = self.notes_manager.num_lengths  # number of note lengths in our input

        # import pdb;pdb.set_trace()

        prediction_array = self._model.predict(prediction_input)[0]  # prediction array from NN

        pitch_prediction = classes[numpy.argmax(prediction_array[num_lengths:]) + num_lengths]
        length_prediction = classes[numpy.argmax(prediction_array[:num_lengths])]
        prediction_encoded = encoder.transform([[pitch_prediction, length_prediction]])

        # return the normal prediction and the one-hot encoded prediction
        return pitch_prediction, length_prediction, prediction_encoded

    def load(self, filename):
        self._model = keras.models.load_model(filename)

    def save(self, filename):
        self._model.save(filename)

    def load_weights(self, filename):
        self._model.fit(self.notes_manager.training_input,
                        self.notes_manager.training_output, epochs=0)
        self._model.load_weights(filename)
