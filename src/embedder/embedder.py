import torch


def bow_embedder(tensor, vocab_length):
    """Takes a tensor and a vocabulary length, and returns the BoW embedding of the tensor"""
    tensor = tensor.long()
    embedding = torch.zeros(vocab_length)
    for index in tensor:
        embedding[index] += 1
    return embedding

# TODO: We probably don't want to use the BoW embedder for this project, and instead we should implement Word2Vec.
