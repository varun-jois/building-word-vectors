# building-word-vectors
To make sure everything runs first ensure you have tensorflow, keras and numpy. If you don't have the following you can install them by running the following on the command line:

`$ pip install numpy`

`$ pip install tensorflow`

`$ pip install keras`

The client code for building word vectors is located in create_word_vectors.py. All you have to do is run it from the command line and follow the instructions. You will have to provide the full path to a text file to create the word vectors. My suggestion is to use any of the text files in the gutenberg corpus like austen-emma.txt. You will also be asked to provide a path for saving the word vector.

You can play a word association game I've created in the file word_game.py. All you have to do is run it from the terminal. You will need the GloVe embeddings for this. Since the smallest GloVe vector is still too large to upload on github, I have provided a subset of the original with a smaller vocabulary but the same weights.
