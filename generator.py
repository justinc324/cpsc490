#!/usr/bin/python3

"""
Functions mainly responsible for generating
notes from the RNN, as well as transcribing
said notes to MIDI.
"""
import numpy
import music21 as m21
import helpers


def generate_notes(neural_network, num_notes):
    """
    Given a MusicRNN, this function:
    - Grabs notes from the MusicRNN's NoteManager object
    - Uses these notes to create a sequence of notes
    - Uses the NoteManager to encode those notes into a sequence
    appropriate for the MusicRNN
    - Grabs the output prediction from the MusicRNN
    - Translates this output to notes via the NoteManager
    - Outputs this sequence of notes

    neural_network: A MusicRNN object to use to generate notes
    num_notes: An integer indicating the number of notes to create

    returns a list of tuples of numerical pitches and durations

    example: generate_notes(myRNN, 2) --> [(34, 1.0), (43.41, .25)]
    """

    training_input = neural_network.notes_manager.training_input
    sequence_length = neural_network.notes_manager.sequence_len  # timesteps/sequence length
    num_features = neural_network.notes_manager.num_features  # num of unique pitches+lengths (classes)

    nums = numpy.random.random_integers(len(training_input) - 1, size=(1, sequence_length))[0]

    pattern = []
    note_output = []

    for i in range(len(nums)):
        pattern.append(training_input[nums[i]][i].tolist())

    for _ in range(num_notes):

        pred_pitch, pred_length, pred_encoded = neural_network.predict_note(numpy.reshape(pattern, (1, sequence_length, num_features)))

        # eliminate "-" sign on duration
        fixed_duration = pred_length
        if fixed_duration[0] == "-":
            fixed_duration = fixed_duration[1:]

        # append the predicted note to the output
        note_output.append((pred_pitch, fixed_duration))

        pattern.append(pred_encoded[0].tolist())  # append the predicted note (encoded) to the sequence
        pattern = pattern[1:]  # bump off the first note in the sequence

    return note_output


def write_to_midi(notes, filename):
    """
    Given a list of tuples of pitches/durations this function
    translates the numerical notation to an actual note, and
    with the duration creates a Music21 Note, Chord, or Rest.
    It appends each of these objects to a Music21 stream, and
    saves the stream as a MIDI file.

    notes: A list of tuples
    filename: A string to save the MIDI file as

    Example:
        write_to_midi((50, 1.0), "my_song" --> my_song.mid
    """
    s1 = m21.stream.Stream()

    for n in notes:

        pitch = n[0].split(",")
        duration = None
        numerator = None

        # convert to float or grab fraction
        try:
            duration = float(n[1])
        except ValueError:
            numerator, denominator = (n[1]).split("/")
            numerator = int(numerator)
            denominator = int(denominator)

        note = None

        if duration:
            if len(pitch) == 1:

                pitch = pitch[0]

                if pitch == "X":
                    note = m21.note.Rest(quarterLength=duration)
                else:
                    pitch = helpers.convert_number(pitch)
                    note = m21.note.Note(pitch, quarterLength=duration)

            else:
                chord_notes = [helpers.convert_number(x) for x in pitch]
                note = m21.chord.Chord(chord_notes, quarterLength=duration)

        elif numerator:
            if len(pitch) == 1:

                pitch = pitch[0]

                if pitch == "X":
                    note = m21.note.Rest(quarterLength=numerator/denominator)
                else:
                    # import pdb; pdb.set_trace()
                    pitch = helpers.convert_number(pitch)
                    note = m21.note.Note(pitch, quarterLength=numerator/denominator)

            else:
                # import pdb; pdb.set_trace()
                chord_notes = [helpers.convert_number(x) for x in pitch]
                note = m21.chord.Chord(chord_notes, quarterLength=numerator/denominator)

        elif duration == 0.0:
            continue

        s1.append(note)

    fp = s1.write('midi', fp=filename)
