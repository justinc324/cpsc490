#!/usr/bin/python3

import nn
import generator as gen
import parse_midi as pm


class SongCreator:

    def __init__(self, genre, instrument, parsed=False, epochs=100, batch_size=32):

        # set genre, instrument, and path
        self.genre = genre
        self.instrument = instrument
        self.path = "{}/{}".format(genre, instrument)

        # parse notes if they haven't already been parsed
        if not parsed:
            self.parse()

        # create NoteManager objects
        self.introManager = pm.NotesManager("{}/parsed_notes/intro.pickle".format(self.path))
        self.middleManager = pm.NotesManager("{}/parsed_notes/middle.pickle".format(self.path))
        self.outroManager = pm.NotesManager("{}/parsed_notes/outro.pickle".format(self.path))

        # grab the training inputs and outputs
        self.intro_training_input, self.intro_training_output = self.introManager.create_sequences()
        self.middle_training_input, self.middle_training_output = self.middleManager.create_sequences()
        self.outro_training_input, self.outro_training_output = self.outroManager.create_sequences()

        # create recurrent neural networks
        self.introRNN = nn.MusicRNN(self.introManager, epochs=epochs, batch_size=batch_size)
        self.middleRNN = nn.MusicRNN(self.middleManager, epochs=epochs, batch_size=batch_size)
        self.outroRNN = nn.MusicRNN(self.outroManager, epochs=epochs, batch_size=batch_size)

    def parse(self):
        """
        Parses the songs in the genre/instrument/training_songs folder and extracts
        the notes to the genre/instrument/parsed_notes folder.
        """
        pm.parse_notes("{}/training_songs/*.mid".format(self.path),
                       "{}/parsed_notes".format(self.path))

    def load_nn(self):
        """
        Loads previously saved neural networks from the
        nn_models folder.
        """
        self.introRNN.load("{}/nn_models/nn_intro".format(self.path))
        self.middleRNN.load("{}/nn_models/nn_middle".format(self.path))
        self.outroRNN.load("{}/nn_models/nn_outro".format(self.path))

    def load_nn_weights(self):
        """
        Loads previously saved neural network weights from
        the nn_weights folder.
        """
        self.introRNN.load_weights("{}/nn_weights/intro.weights.best.hdf5".format(self.path))
        self.middleRNN.load_weights("{}/nn_weights/middle.weights.best.hdf5".format(self.path))
        self.middleRNN.load_weights("{}/nn_weights/middle.weights.best.hdf5".format(self.path))

    def train(self, save_weights=True):
        """
        Trains the neural networks. Saves
        weights by default to genre/instrument/nn_weights
        as '().weights.best.hdf5'

        save_weights: A boolean
        """

        # trains and save the neural networks
        if save_weights:
            self.introRNN.train(self.intro_training_input, self.intro_training_output,
                                filename="{}/nn_weights/intro".format(self.path))
            self.middleRNN.train(self.middle_training_input, self.middle_training_output,
                                 filename="{}/nn_weights/middle".format(self.path))
            self.outroRNN.train(self.outro_training_input, self.outro_training_output,
                                filename="{}/nn_weights/outro".format(self.path))
        # train without saving
        else:
            self.introRNN.train(self.intro_training_input, self.intro_training_output)
            self.middleRNN.train(self.middle_training_input, self.middle_training_output)
            self.outroRNN.train(self.outro_training_input, self.outro_training_output)

    def save_model(self):
        """
        Saves the models of the neural networks
        to genre/instrument/nn_models
        """
        self.introRNN.save("{}/nn_models/nn_intro".format(self.path))
        self.middleRNN.save("{}/nn_models/nn_middle".format(self.path))
        self.outroRNN.save("{}/nn_models/nn_outro".format(self.path))

    def create_song(self, song_name, intro_len=32, verse_len=52, chorus_len=64,
                    bridge_len=48, outro_len=32):
        """
        Generates notes from the intro, middle, and outro
        RNNs, combining them to create a single song. It
        is then converted to MIDI and saved. Saves the
        song under genre/instrument/nn_songs.

        structure: intro->verse->chorus->verse->chorus->bridge->chorus_outro

        song_name: A string, name of the file to save the song as
        intro_len: An integer, the length of the intro of the song
        verse_len: An integer, the length of the verse of the song
        chorus_len: An integer, the length of the chorus of the song
        bridge_len: An integer, the length of the bridge of the song
        outro_len: An integer, length of the outro of the song
        """

        intro_notes = gen.generate_notes(self.introRNN, intro_len)
        verse1_notes = gen.generate_notes(self.middleRNN, verse_len)
        chorus_notes = gen.generate_notes(self.middleRNN, chorus_len)
        verse2_notes = gen.generate_notes(self.middleRNN, verse_len)
        bridge_notes = gen.generate_notes(self.middleRNN, bridge_len)
        outro_notes = gen.generate_notes(self.outroRNN, outro_len)

        song_combined = (intro_notes + verse1_notes + chorus_notes + verse2_notes +
                         chorus_notes + bridge_notes + chorus_notes + outro_notes)

        gen.write_to_midi(song_combined, "{}/nn_songs/{}".format(self.path, song_name))
