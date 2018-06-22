# PyTorch Implementation of Variational Recurrent Autoencoder #

This is a well-structured VRAE implementation for future research uses. It has been tested on Pytorch 4.0 and Python 3.6.
An example dataset located in `examples/`. Use this to try training and
generating samples.

To train, run the following command from the root repository:

    python -m train --data-path examples/sents.txt --save-dir examples/
    
To generate, run the following command:

    python -m generate --vocab examples/vocab.pkl -ckpt-path examples/checkpoint-e01 --save-dir examples/
