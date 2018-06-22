# PyTorch Implementation of Variational Recurrent Autoencoder #

This is a well-structured VRAE implementation for future research use. 
An example dataset located in `examples/`. Use this to try training and
generating samples.

To train, run the following command from the root repository:

    python -m test --data-path examples/sents.txt --save-dir examples/
    
To generate, run the following command:

    python -m generate --vocab examples/vocab.pkl -ckpt-path examples/checkpoint-e01 --save-dir examples/