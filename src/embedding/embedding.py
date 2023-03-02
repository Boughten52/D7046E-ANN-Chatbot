from torch.utils.data import Dataset
import numpy as np


class ChatBotDataset(Dataset):
    def __init__(self, x, y):
        self.samples = len(x)
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.samples


def bow_embedder(tokenized_sentence, vocab):
    bow = np.zeros(len(vocab), dtype=np.float32)
    for index, word in enumerate(vocab):
        if word in tokenized_sentence:
            bow[index] = 1
    return bow
