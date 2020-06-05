import re
import csv
import pickle
import numpy as np
from nltk.corpus import stopwords


RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'model/intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'model/tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'model/tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'embedding_of_differ_tags',
    'WORD_EMBEDDINGS': 'stackoverflow_out_model.tsv'
}

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = GOOD_SYMBOLS_RE.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in STOPWORDS])
    return text.strip()


def array_to_string(arr):
    return '\n'.join(str(num) for num in arr)


def matrix_to_string(matrix):
    return '\n'.join('\t'.join(str(num) for num in line) for line in matrix)


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    embeddings = {}
    with open(embeddings_path, newline='') as embedding_file:
        reader = csv.reader(embedding_file, delimiter='\t')
        for line in reader:
            word = line[0]
            embedding = np.array(line[1:]).astype(np.float32)
            embeddings[word] = embedding
        dim = len(line)-1
    return embeddings, dim


def question_to_vec(question, embeddings, dim=100):
    words_embedding = [embeddings[word] for word in question.split() if word in embeddings]
    if not words_embedding:
        return np.zeros(dim)
    words_embedding = np.array(words_embedding)
    return words_embedding.mean(axis=0)


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)