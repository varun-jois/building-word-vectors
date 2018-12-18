
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, Reshape, merge, dot
from keras.preprocessing.sequence import make_sampling_table, skipgrams
import numpy as np


class WordVector:
    def __init__(self, text, vocabulary_size=None, vector_dimensions=10, window_size=5, iterations=100000):
        self.text = text
        self.vocabulary_size = vocabulary_size
        self.vector_dimensions = vector_dimensions
        self.window_size = window_size
        self.iterations = iterations
        self.model = None

    def _tokenize(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts=[self.text])
        tokens_indices = tokenizer.texts_to_sequences(texts=[self.text])[0]
        types = max(tokens_indices)
        if self.vocabulary_size is None:
            if types <= 10000:
                self.vocabulary_size = types
            else:
                self.vocabulary_size = 10000
                tokens_indices = [0 if i > 10000 else i for i in tokens_indices]
        return tokens_indices

    def _get_word_pairs(self, tokens_indices):
        sampling_table = make_sampling_table(size=self.vocabulary_size + 1)
        word_pairs, labels = skipgrams(sequence=tokens_indices, vocabulary_size=self.vocabulary_size,
                                       window_size=self.window_size, sampling_table=sampling_table)
        return word_pairs, labels

    def _build_model(self):
        target = Input((1,))
        context = Input((1,))
        word_vector = Embedding(input_dim=self.vocabulary_size + 1, output_dim=self.vector_dimensions, input_length=1)
        target_vector = word_vector(target)
        target_vector = Reshape((self.vector_dimensions, 1))(target_vector)
        context_vector = word_vector(context)
        context_vector = Reshape((self.vector_dimensions, 1))(context_vector)
        # output = Dense(1, activation='sigmoid')(dot(inputs=[target_vector, context_vector], axes=0))
        dot_product = dot(inputs=[target_vector, context_vector], axes=1)
        # dot_product = merge.dot([target_vector, context_vector], axes=1)
        dot_product = Reshape((1,))(dot_product)
        output = Dense(1, activation='sigmoid')(dot_product)
        model = Model(input=[target, context], output=output)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self):
        tokens_indices = self._tokenize()
        word_pairs, labels = self._get_word_pairs(tokens_indices=tokens_indices)
        self.model = self._build_model()
        target = np.zeros((1,))
        context = np.zeros((1,))
        label = np.zeros((1,))
        for i in range(self.iterations):
            index = np.random.randint(0, len(labels) - 1)
            target[0, ], context[0, ] = word_pairs[index]
            label[0, ] = labels[index]
            loss = self.model.train_on_batch([target, context], label)
            if i % 5000 == 0:
                print('iteration, loss = %d, %d' % (i, loss))


