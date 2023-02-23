from src import dataloader


class AmazonVocabulary:
    """Class that contains a vocabulary as two dictionaries (token2index, index2token)"""

    def __init__(self):
        """Creates a vocabulary of the Amazon reviews"""
        loader = dataloader.AmazonDataLoader("dataloader/amazon_cells_labelled.txt")
        amazon_vocabulary = loader.vocabulary
        self.token2index = {}
        self.index2token = {}
        for token in amazon_vocabulary.keys():
            index = amazon_vocabulary[token]
            self.token2index[token] = index
            self.index2token[index] = token

    def get_index(self, token):
        """Takes a token and returns its index"""
        if token not in self.token2index:
            return None
        else:
            return self.token2index[str(token).lower()]

    def get_token(self, index):
        """Takes an index and returns its token"""
        return self.index2token[int(index)]

    def __len__(self):
        """Returns the length of the vocabulary"""
        return len(self.token2index)
