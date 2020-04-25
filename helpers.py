#!/usr/bin/python3

import music21 as m21
import glob

# dictionary converting notes to numbers
note_to_num_conversion = {
    "B#" : 0,
    "C" : 0,
    "C#" : 1,
    "D-" : 1,
    "D" : 2,
    "D#" : 3,
    "E-" : 3,
    "E" : 4,
    "F-" : 4,
    "F" : 5,
    "E#" : 5,
    "F#" : 6,
    "G-" : 6,
    "G" : 7,
    "G#" : 8,
    "A-" : 8,
    "A" : 9,
    "A#" : 10,
    "B-" : 10,
    "B" : 11,
    "C-": 11
}

# dictionary converting numbers to notes
num_to_note_conversion = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B"
}

# major conversions
majors = dict([("A-", 4),("G#", 4),("A", 3),("A#", 2),("B-", 2),("B", 1),("C", 0),("C#", -1),("D-", -1),("D", -2),
               ("D#", -3),("E-", -3),("E", -4),("F", -5),("F#", 6),("G-", 6),("G", 5)])
minors = dict([("G#", 1), ("A-", 1),("A", 0),("A#", -1),("B-", -1),("B", -2),("C", -3),("C#", -4),("D-", -4),("D", -5),
               ("D#", 6),("E-", 6),("E", 5),("F", 4),("F#", 3),("G-", 3),("G", 2)])


def transpose_songs(path):

    for file in glob.glob(path):
        print(file)
        score = m21.converter.parse(file)
        key = score.analyze('key')

        if key.mode == "major":
            halfSteps = majors[key.tonic.name]

        elif key.mode == "minor":
            halfSteps = minors[key.tonic.name]

        newscore = score.transpose(halfSteps)
        key = newscore.analyze('key')
        print(key.tonic.name, key.mode)
        newFileName = file.rsplit('/',1)[0] + "/" + "C_" + file.rsplit('/',1)[1]
        newscore.write('midi', newFileName)


def convert_note(note):
    """
    converts a given note to a numerical representation
    C0 --> 0
    D4 --> 50
    """

    # grab the octave
    octave = int(note[-1])

    # grab the pitch
    if len(note) > 2:
        pitch = note[0:2]
    else:
        pitch = note[0]

    # convert the pitch and octave to the correct number
    return note_to_num_conversion[pitch] + (octave * 12)


def convert_number(number):
    """
    converts a given number to its note representation
    0 --> C0
    50 -- > D4
    """
    number = int(number)
    pitch = (number % 12)
    octave = (number - pitch) // 12
    return "{}{}".format(num_to_note_conversion[pitch], octave)