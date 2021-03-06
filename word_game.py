
"""This program plays the word association game with the user.

"""

from WordVector import WordVector
from random import shuffle


def load_model():
    """This function loads the GloVe word vectors.

    :return: The word vectors.
    """
    wv = WordVector()
    wv.build_from_existing('glove_modified.txt')
    return wv


def load_words():
    """Loads the predefined words for the game.

    :return: The words.
    """
    with open('word_game_words.txt', 'r') as f:
        lines = f.readlines()
    shuffle(lines)
    words = [l.split() for l in lines]
    return words


def play_round(score, word_vector, word1, word2, word3, word4):
    """Plays a single round of the game.

    :param list score: The current scores in the form [user score, comp score]
    :param WordVector word_vector: The GloVe vector.
    :param str word1: A word that's used for the game. A context word.
    :param str word2: A word that's used for the game. A context word.
    :param str word3: A word that's used for the game. A context word.
    :param str word4: A word that's used for the game. The target word.
    :return:
    """
    user_input = input('%s: %s :: %s: ' % (word1, word2, word3))
    comp_input = word_vector.find_best_relationship(word1, word2, word3)[0]
    print('The correct answer is: ' + word4)
    print('The vector guessed: ' + comp_input)
    if user_input == word4 and comp_input == word4:
        print('You both got it right! +1 for both!')
        score[0] += 1
        score[1] += 1
    elif user_input == word4 and comp_input != word4:
        print('You beat the vector! +1 for you.')
        score[0] += 1
    elif user_input != word4 and comp_input == word4:
        print('The vector beat you this time! +1 for the vector.')
        score[1] += 1
    else:
        print('You both got it wrong :(')


if __name__ == '__main__':
    print('\n\nWelcome to a game of `Can you beat a super smart vector!`')
    print('You\'re job is to find that correct word that maps the association.')
    print('For example, king: man :: woman: queen')
    print('Another example, gates: microsoft :: apple: jobs')
    print('Let\'s see if you can beat the vector!\n')
    wv = load_model()
    words = load_words()
    score = [0, 0]
    for w in words:
        play_round(score, wv, *w)
        print()
    print('That\'s the end of the game! I hope you enjoyed it. Final scores are: ')
    print('|%-10s|%-10s|' % ('User', 'Vector'))
    print('|%-10d|%-10d|' % (score[0], score[1]))
