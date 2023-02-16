# TODO: This vocabulary is the same as in ANN exercise 2. It's not yet compatible with the dataloader for
#       amazon_cells_labelled.txt.
class Vocabulary:
    """Class that contains a vocabulary as two dictionaries (token2index, index2token)"""

    def __init__(self, corpus):
        """Takes a corpus and tokenizes it"""
        current_index = 0
        self.token2index = {}
        self.index2token = {}
        for text in corpus:
            tokens = text.split()
            for token in tokens:
                if token not in self.token2index:
                    self.token2index[token] = current_index
                    self.index2token[current_index] = token
                    current_index += 1

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
