import torch


# If you want a function or class to be available in another Python file you type "from .embedder import NAME" in the
# __init__.py file in this directory.
# In the file where you want to use it you type "import embedder", and to then for example call a function
# you type "embedder.FUNCTION_NAME()".
def bow_embedder(tensor, vocab_length):
    """Takes a tensor and a vocabulary length, and returns the BoW embedding of the tensor"""
    tensor = tensor.long()
    embedding = torch.zeros(vocab_length)
    for index in tensor:
        embedding[index] += 1
    return embedding

# TODO: We probably don't want to use the BoW embedder for this project, and instead we should implement Word2Vec.
