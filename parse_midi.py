#!/usr/bin/python3

from helpers import *
import pickle
import numpy
from sklearn.preprocessing import MultiLabelBinarizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NotesManager:

    def __init__(self, file, sequence_len=16):

        self._notes = self.__load_notes__(file)
        self.pitches, self.lengths = self.__separate_notes__(self._notes)
        self.num_features = None  # number of unique notes/chords
        self.sequence_len = sequence_len

        self.one_hot_encoder = MultiLabelBinarizer()  # translates input/output to the NN

        self.training_input = None
        self.training_output = None

    @staticmethod
    def __load_notes__(file):
        """
        loads notes from the given file via pickling
        """
        # grab the notes from the given file
        with open(file, "rb") as fp:
            notes = pickle.load(fp)

        return notes

    def __separate_notes__(self, notes):
        """
        Separates the given notes into pitches and length
        """

        # separate the pitches and lengths of the notes
        pitches = []
        lengths = []

        # separate the notes from "10.2:.25" --> ["10.2", ".25"]
        for n in notes:
            pitches.append(n[0])
            lengths.append(n[1])

        self.num_pitches = len(set(pitches))
        self.num_lengths = len(set(lengths))

        return pitches, lengths

    def create_sequences(self):
        """
        creates input and output sequences to pass to the neural network
        If the sequences have already been created, return the sequences.
        """

        # if data is already available, return the given data instead of calculating it again
        if self.training_input and self.training_output:
            return self.training_input, self.training_output

        training_input = []
        training_output = []

        # translator for encoding/decoding input and output to the NN
        notes = self.one_hot_encoder.fit_transform(self._notes)

        for i in range(len(notes) - self.sequence_len):

            training_input.append([x for x in notes[i:i+self.sequence_len]])
            training_output.append(notes[i+self.sequence_len])

        # transform to numpy array
        training_input = numpy.asarray(training_input)
        training_output = numpy.asarray(training_output)

        # save the training input/output and encoder to the class
        self.num_features = len(self.one_hot_encoder.classes_)
        self.training_input = training_input
        self.training_output = training_output

        return training_input, training_output


def parse_notes(fp_songs, fp_out, intro_split=24, outro_split=24):
    """
    fp_songs: path to the folder of songs to parse notes from
    fp_out: filepath of the folder to save the notes
    intro_split: number of notes to take from the beginning of the
    song for the intro folder
    outro_split: number of notes to take from the end of the song
    for the outro folder
    """

    intro_notes = []
    middle_notes = []
    outro_notes = []

    for file in glob.glob(fp_songs):

        song_notes = []

        try:
            song = m21.converter.parse(file)
        except:
            print("could not parse file")
            continue

        notes_to_parse = song.flat.notes

        for element in notes_to_parse:
            duration = element.duration.quarterLength
            note = None
            # find out if the element is a rest, note, or chord
            if isinstance(element, m21.note.Rest):

                note = "X"
            elif isinstance(element, m21.note.Note):
                note = (str(convert_note(str(element.pitch))))
            elif isinstance(element, m21.chord.Chord):

                note = ','.join(str(convert_note(str(n.pitch))) for n in element)
            else:
                continue

            # append the note and its duration to the list
            song_notes.append([note, "-" + str(duration)])

        last_index = len(song_notes) - 1

        # split the song notes
        intro_notes.extend(song_notes[:intro_split])
        middle_notes.extend(song_notes[intro_split:last_index-outro_split])
        outro_notes.extend(song_notes[last_index - outro_split:])

    # file paths and names for the parsed notes
    intro_fp = "{}/intro.pickle".format(fp_out)
    middle_fp = "{}/middle.pickle".format(fp_out)
    outro_fp = "{}/outro.pickle".format(fp_out)

    # this uses the python pickle module to write it as a list
    with open(intro_fp, "wb") as fw:
        pickle.dump(intro_notes, fw)

    with open(middle_fp, "wb") as fw:
        pickle.dump(middle_notes, fw)

    with open(outro_fp, "wb") as fw:
        pickle.dump(outro_notes, fw)