#!/usr/bin/python3

import sys
import getopt
import song_creator as sc


def main(argv):

    helpstring = ("main.py -t -g <song_name> -m <music genre> -i <instrument> -s <save> -l <load>\n"
                  "-t: trains a neural network for the specified music/instrument\n"
                  "-g: generates music for the specified music/instrument as the given song name\n"
                  "-m: music genre to use\n"
                  "-i: instrument to use\n"
                  "-s: save the NN model in music/instrument/nn_model\n"
                  "-l: load a previously saved NN model in music/instrument/nn_model")

    train = False
    generate = None
    music = None
    instrument = None
    save = None
    load = None

    try:
        opts, args = getopt.getopt(argv, "tg:m:i:sl", ["train", "generate=", "music=", "instrument=", "save", "load"])
    except getopt.GetoptError:
        print(helpstring)
        sys.exit(2)

    if len(opts) == 0:
        print(helpstring)
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print(helpstring)
            sys.exit()
        elif opt in ("-t", "--train"):
            train = True
        elif opt in ("-g", "--generate"):
            generate = arg
        elif opt in ("-m", "--music"):
            music = arg
        elif opt in ("-i", "--instrument"):
            instrument = arg
        elif opt in ("-s", "--save"):
            save = arg
        elif opt in ("-l", "--load"):
            load = arg

    # error checking
    if train or generate:
        if not music or not instrument:
            print("-m and -i args required when training or generating")
    if load:
        if train:
            print("-l option not valid when training")
    if save:
        if not train:
            print("-t option required to save a NN")

    # nn = sc.SongCreator(music, instrument, parsed=False, epochs=200, batch_size=32)
    nn = sc.SongCreator(music, instrument, parsed=False, epochs=5, batch_size=256)

    # train neural network if that was selected as an option
    if train:
        nn.train()

        # save the model if indicated
        if save:
            nn.save_model()

    # load a previous model
    elif load:
        nn.load_nn()

    # generate if indicated
    if generate:

        # if not trained or loaded, load the weights
        if not train and not load:
            nn.load_nn_weights()

        # make the song
        nn.create_song(generate + ".mid")

        print("Song created under {}/{}/nn_songs/{}.mid".format(music, instrument, generate))

    # exit
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
