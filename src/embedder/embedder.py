import torch
import re


def bow_embedder(text, vocabulary):
    """Takes any sentence and vocabulary with the token2index and index2token structure, returns the BoW embedding"""

    # Remove special characters, make lower case and split sentence
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    text = text.lower()
    text = text.split(" ")

    unrecognized_words = []

    embedding = torch.zeros(len(vocabulary))
    for word in text:
        if word in vocabulary.token2index.keys():
            embedding[vocabulary.token2index[word]] += 1
        else:
            unrecognized_words.append(word)

    if len(unrecognized_words) > 0:
        print("When embedding, these words were not recognized:", unrecognized_words)

    return embedding
