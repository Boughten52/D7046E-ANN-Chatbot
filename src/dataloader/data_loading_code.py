import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize


def preprocess_pandas(_data, _columns):
    df_ = pd.DataFrame(columns=_columns)
    _data['Sentence'] = _data['Sentence'].str.lower()
    _data['Sentence'] = _data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)  # remove emails
    _data['Sentence'] = _data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '',
                                                  regex=True)  # remove IP address
    _data['Sentence'] = _data['Sentence'].str.replace('[^\w\s]', '', regex=True)  # remove special characters
    _data['Sentence'] = _data['Sentence'].replace('\d', '', regex=True)  # remove numbers

    # THIS CODE UPDATES A VARIABLE THAT IS NEVER USED
    # for index, row in _data.iterrows():
    #     word_tokens = word_tokenize(row['Sentence'])
    #     filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
    #     df_ = df_.append({
    #         "index": row['index'],
    #         "Class": row['Class'],
    #         "Sentence": " ".join(filtered_sent[0:])
    #     }, ignore_index=True)
    return _data


class DataLoader:
    """The code from Canvas for loading a set of Amazon reviews has been placed in a class"""

    def __init__(self, file_path):
        # get data, pre-process and split
        data = pd.read_csv(file_path, delimiter='\t', header=None)
        data.columns = ['Sentence', 'Class']
        data['index'] = data.index  # add new column index
        columns = ['index', 'Class', 'Sentence']
        self.data = preprocess_pandas(data, columns)  # pre-process
        training_data, validation_data, training_labels, validation_labels = train_test_split(
            # split the data into training, validation, and test splits
            data['Sentence'].values.astype('U'),
            data['Class'].values.astype('int32'),
            test_size=0.10,
            random_state=0,
            shuffle=True
        )

        # vectorize data using TFIDF and transform for PyTorch for scalability
        word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=50000, max_df=0.5,
                                          use_idf=True,
                                          norm='l2')
        training_data = word_vectorizer.fit_transform(training_data)  # transform texts to sparse matrix
        self.training_data = training_data.todense()  # convert to dense matrix for Pytorch
        validation_data = word_vectorizer.transform(validation_data)
        self.validation_data = validation_data.todense()

        self.vocabulary = word_vectorizer.vocabulary_
        self.vocab_size = len(self.vocabulary)
        self.train_x_tensor = torch.from_numpy(np.array(self.training_data)).type(torch.FloatTensor)
        self.train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
        self.validation_x_tensor = torch.from_numpy(np.array(self.validation_data)).type(torch.FloatTensor)
        self.validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()


if __name__ == '__main__':
    loader = DataLoader("amazon_cells_labelled.txt")
