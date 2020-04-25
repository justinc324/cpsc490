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
        # self.pitch_to_int, self.int_to_pitch = self.__encode_pitches__(self.pitches)
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

        return pitches, lengths,

    @staticmethod
    def __encode_pitches__(pitches):
        """
        Returns a dictionary encoding pitches to an integer,
        and vice-versa
        """
        int_to_pitch = {k: v for k, v in enumerate(sorted(set(pitches)))}
        pitch_to_int = {v: k for k, v in enumerate(sorted(set(pitches)))}

        return pitch_to_int, int_to_pitch

    def create_sequences(self):
        """
        creates input and output sequences to pass to the neural network
        If the sequences have already been created, return the sequences.
        """

        # # a list of sequences of notes and pitches
        # pitch_input = []
        # length_input = []
        #
        # # a list of notes that follow the previous sequence (one to one mapping)
        # pitch_output = []
        # length_output = []
        #
        # # loop through self.pitches and self.lengths and grab sequences of pitches/lengths,
        # # while also integer-converting pitches and 0-1 normalizing the length
        # for i in range(0, len(self.pitches) - self.sequence_len):
        #
        #     pitch_input.append([self.pitch_to_int[x] for x in self.pitches[i:i + self.sequence_len]])
        #     length_input.append([(x/self._longest_note) for x in self.lengths[i:i + self.sequence_len]])
        #
        #     pitch_output.append(self.pitch_to_int[self.pitches[i+self.sequence_len]])
        #     length_output.append((self.lengths[i+self.sequence_len]/self._longest_note))

        # if data is already available, return the given data instead of calculating it again
        if self.training_input and self.training_output:
            return self.training_input, self.training_output

        training_input = []
        training_output = []

        # import pdb; pdb.set_trace()

        # translator for encoding/decoding input and output to the NN
        notes = self.one_hot_encoder.fit_transform(self._notes)

        for i in range(len(notes) - self.sequence_len):

            training_input.append([x for x in notes[i:i+self.sequence_len]])
            training_output.append(notes[i+self.sequence_len])

        # transform to numpy array
        training_input = numpy.asarray(training_input)
        training_output = numpy.asarray(training_output)


        # one hot encode the note_input sequences
        # input_encoded = numpy.asarray(pitch_input)
        # input_encoded = keras.utils.to_categorical(input_encoded)
        #
        # # append the normalized lengths to the one-hot encoded input pitches
        # input_encoded = input_encoded.tolist()
        # for i in range(0, len(pitch_input)):
        #     for j in range(0, len(pitch_input[i])):
        #         input_encoded[i][j].append(length_input[i][j])

        # # one-hot encode the output pitches
        # output_encoded = numpy.asarray(pitch_output)
        # output_encoded = keras.utils.to_categorical(output_encoded)
        #
        # # append the normalized output lengths to the one-hot encoded output pitches
        # output_encoded = output_encoded.tolist()
        # for i in range(len(output_encoded)):
        #     output_encoded[i].append(length_output[i])
        #
        # # output as numpy arrays
        # input_encoded = numpy.asarray(input_encoded)
        # output_encoded = numpy.asarray(output_encoded)

        # import pdb; pdb.set_trace()

        # save the training input/output and encoder to the class
        self.num_features = len(self.one_hot_encoder.classes_)
        self.training_input = training_input
        self.training_output = training_output

        return training_input, training_output

    def translate_note(self, note):
        """
        Takes a list or numpy array and translates and returns the original pitch
        and length.
        Pitch: One-hot encoding --> label encoding --> original pitch
        Note: Normalized length --> original length
        input: [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.375]
        output: ['42.35', 0.75]
        """
        # convert pitch from one hot-encoding and then label encoding to the original pitch
        pitch = self.int_to_pitch[numpy.argmax(note[0:self.num_notes-1])]

        # convert pitch from normalization
        length = note[self.num_notes] * self._longest_note

        return pitch, length


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

        song = m21.converter.parse(file)

        # parts = m21.instrument.partitionByInstrument(song)
        #
        # if parts:
        #     notes_to_parse = parts.parts[0].recurse()
        # else:
        notes_to_parse = song.flat.notes

        for element in notes_to_parse:
            duration = element.duration.quarterLength
            note = None
            # find out if the element is a rest, note, or chord
            if isinstance(element, m21.note.Rest):
                # import pdb;pdb.set_trace()
                note = "X"
            elif isinstance(element, m21.note.Note):
                note = (str(convert_note(str(element.pitch))))
            elif isinstance(element, m21.chord.Chord):
                # import pdb; pdb.set_trace()
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


def parse_notes_no_split(fp_songs, fp_out):

    notes = []

    for file in glob.glob(fp_songs):

        song_notes = []

        song = m21.converter.parse(file)

        parts = m21.instrument.partitionByInstrument(song)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = song.flat.notes

        for element in notes_to_parse:
            duration = element.duration.quarterLength
            note = None
            # find out if the element is a rest, note, or chord
            if isinstance(element, m21.note.Rest):
                # import pdb;pdb.set_trace()
                note = "X"
            elif isinstance(element, m21.note.Note):
                note = (str(convert_note(str(element.pitch))))
            elif isinstance(element, m21.chord.Chord):
                # import pdb; pdb.set_trace()
                note = ','.join(str(convert_note(str(n.pitch))) for n in element)
            else:
                continue

            # append the note and its duration to the list
            notes.append([note, "-" + str(duration)])

    # this uses the python pickle module to write it as a list
    with open(fp_out, "wb") as fw:
        pickle.dump(notes, fw)

    return notes
# notes_read = preprocessing("notes/piano.txt")


# if notes == notes_read:
#     print("gucci")
