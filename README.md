# textgenrnn-torch
A simple text generator using either a word-level or a character-level LSTM.


## Usage
The required arguments are the mode (`train` or `generate`) and `--model` to
specify the model file. Training also requires `--input`.

### Training
To train a simple word-level network with default values, use

    textgenrnn-torch.py train --model model.torch --input input.txt

To train a character-level network, add `--token-level char`:

    textgenrnn-torch.py train --model model.torch --input input.txt --token-level char

The script automatically uses different default parameters for the
embedding dimension, the hidden dimension, and the maximum sequence length for
word-level and character-level networks. All network parameters can also be
changed using command line options, run `textgenrnn-torch.py --help` for the
full list of parameters.

### Generation
To generate text run

    textgenrnn-torch.py generate --model model.torch

You can use `--min-distance` and `--compare-file` to filter out sentences that
are close to sentences in the input data. If `--include-data` was set when
training the network, `--min-distance` will use the included data if no other
file is specified with `--compare-file`.

## Acknowledgements
Thanks to [textgenrnn](https://github.com/minimaxir/textgenrnn/) for the
inspiration and the original Tensorflow implementation.

Note that while this script uses the same ideas, the models are not compatible
and the two implementations have some different features.
