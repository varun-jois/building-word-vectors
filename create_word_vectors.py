
"""This is the client code for generating word vectors on a text. Make sure you have tensorflow,
   keras and numpy installed.

"""
from WordVector import WordVector

VECTOR_DIMENSIONS = 10
WINDOW_SIZE = 3
ITERATIONS = 50000

if __name__ == '__main__':
    print('\n\nThis program builds word vectors from a given .txt file and saves it to disk.')
    fpath = input('Please provide the full path of the .txt file: ')
    user_input = input('Would you like to proceed with the default settings? (y/n): ')
    if user_input == 'n':
        VECTOR_DIMENSIONS = int(input('What size vector would you like? (int > 0): '))
        WINDOW_SIZE = int(input('What window size would you like? (int > 0): '))
        ITERATIONS = int(input('How many iterations would you like to run? (int > 0): '))
    with open(fpath, 'r') as f:
        text = f.read()
    wv = WordVector()
    print('Starting the training.......')
    wv.train(text=text, vector_dimensions=VECTOR_DIMENSIONS, window_size=WINDOW_SIZE, iterations=ITERATIONS)
    fpath = input('Please provide the full path of the .txt file where you would like to save the weights: ')
    wv.save_vector_to_txt(fpath)
