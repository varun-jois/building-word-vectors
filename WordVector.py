
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, Reshape, dot
from keras.preprocessing.sequence import make_sampling_table, skipgrams
from json import loads
import numpy as np


class WordVector:
    def __init__(self):
        self.text = None
        self.vocabulary_size = None
        self.vector_dimensions = None
        self.window_size = None
        self.iterations = None
        self.word_vector = None
        self.word_to_index = None
        self.index_to_word = None

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
        self.word_to_index = loads(tokenizer.get_config()['word_index'])
        self.index_to_word = loads(tokenizer.get_config()['index_word'])
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
        model = Model(inputs=[target, context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, text, vocabulary_size=None, vector_dimensions=10, window_size=5, iterations=100000):
        self.text = text
        self.vocabulary_size = vocabulary_size
        self.vector_dimensions = vector_dimensions
        self.window_size = window_size
        self.iterations = iterations
        tokens_indices = self._tokenize()
        word_pairs, labels = self._get_word_pairs(tokens_indices=tokens_indices)
        model = self._build_model()
        target = np.zeros((1,))
        context = np.zeros((1,))
        label = np.zeros((1,))
        for i in range(self.iterations):
            index = np.random.randint(0, len(labels) - 1)
            target[0, ], context[0, ] = word_pairs[index]
            label[0, ] = labels[index]
            loss = model.train_on_batch([target, context], label)
            if i % 5000 == 0:
                print('iteration: %d' % i)
        self.word_vector = model.get_layer(index=2).get_weights()[0]

    def build_from_existing(self, fpath):
        word_to_index = {}
        index_to_word = {}
        vectors = []
        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                contents = line.split()
                word_to_index[contents[0]] = i + 1
                index_to_word[str(i + 1)] = contents[0]
                vectors.append([float(e) for e in contents[1:]])
        vectors.insert(0, [0] * len(vectors[0]))
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        self.word_vector = np.array(vectors)

    @staticmethod
    def _cosine_similarity(a, b):
        return a.dot(b.T) / np.linalg.norm(a) / np.linalg.norm(b)

    def get_cosine_similarity(self, word1, word2):
        word1_index = self.word_to_index[word1]
        word2_index = self.word_to_index[word2]
        word1_vector = self.word_vector[word1_index]
        word2_vector = self.word_vector[word2_index]
        return self._cosine_similarity(word1_vector, word2_vector)

    def get_similar_words(self, word, n=5, reverse=False):
        target_word_index = self.word_to_index[word]
        similarity_scores = [(i, self._cosine_similarity(self.word_vector[target_word_index], self.word_vector[i]))
                             for i in range(1, self.word_vector.shape[0]) if target_word_index != i]
        similarity_scores.sort(key=lambda x: x[1], reverse=not reverse)
        similar_words = [self.index_to_word[str(i[0])] for i in similarity_scores[:n]]
        return similar_words

    def find_best_relationship(self, word1, word2, word3, n=5):
        word1_index = self.word_to_index[word1]
        word2_index = self.word_to_index[word2]
        word3_index = self.word_to_index[word3]
        target_vector = self.word_vector[word1_index] - self.word_vector[word2_index] + self.word_vector[word3_index]
        similarity_scores = [(i, self._cosine_similarity(target_vector, self.word_vector[i]))
                             for i in range(1, self.word_vector.shape[0])
                             if i not in (word1_index, word2_index, word1_index)]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        similar_words = [self.index_to_word[str(i[0])] for i in similarity_scores[:n]]
        return similar_words

    def save_vector_to_txt(self, fpath):
        with open(fpath, 'w') as f:
            for i in range(1, len(self.index_to_word) + 1):
                word = self.index_to_word[str(i)]
                vector = [str(v) for v in self.word_vector[i]]
                f.write(' '.join([word] + vector) + '\n')
