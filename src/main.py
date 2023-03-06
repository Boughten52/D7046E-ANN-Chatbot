import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize

import chatbot
import embedding
import ann
import nltk
import model_tester


def train():
    """Trains the ANN on the Amazon reviews dataset (expanded with greetings, goodbyes and thanks)"""

    data = []  # Contains sentence and class pairs (sentence as list of words)
    vocab = set()

    crow=0
    with open("data/coursera_pre_strat_train_1800.csv", "r", encoding="UTF-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            crow = crow + 1
            if crow % 100 == 0:
                print(f'Rows read: [{crow}]')
            message, label = row
            message = word_tokenize(message)
            message = [word.lower() for word in message if word.isalnum()]
            data.append((message, int(label)))
            vocab.update(message)

    vocab = list(dict.fromkeys(vocab))
    vocab = sorted(list(vocab))  # Contains all words from the file as single words and without duplicates
    all_labels = {
        0: '1 star',
        1: '2 stars',
        2: '3 stars',
        3: '4 stars',
        4: '5 stars'
    }

    x_train = []
    y_train = []
    ti = 0
    for (tokenized_sentence, label) in data:
        ti = ti + 1
        if ti % 1000 == 0:
            print(f'Tokenized sentences: [{ti}]')
        bow = embedding.bow_embedder(tokenized_sentence, vocab)
        x_train.append(bow)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    dataset = embedding.ChatBotDataset(x_train, y_train)

    # Hyperparameters
    batch_size = 50
    shuffle = True
    input_size = len(vocab)
    hidden_size = 200
    output_size = len(all_labels)
    learning_rate = 0.1
    epochs = 20

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ann.ChatBotANN(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

    loss = 0
    for epoch in range(epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device, dtype=torch.int64)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: [{epoch + 1}/{epochs}] - Loss: {loss}')

    save_data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "vocab": vocab,
        "all_labels": all_labels
    }

    FILE = "chatbot_ann_coursera.pth"
    torch.save(save_data, FILE)


def run():
    """First asks the user if it wants to train the network or chat, then starts the chat-loop"""

    print("Type [bot] to start the chatbot or [train / test] to train/test the network:")
    while True:
        user_input = input()
        if user_input.lower() == "train":
            train()
            return
        if user_input.lower() == "test":
            mytester = model_tester.tester()
            mytester.run_test()
            return
        elif user_input.lower() == "bot":
            break

    bot = chatbot.Chatbot()
    bot.print_response()
    while True:
        bot.handle_input()


if __name__ == '__main__':
    run()
