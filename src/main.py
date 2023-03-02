import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize

import chatbot
import embedding
import ann


def train():
    data = []  # Contains sentence and class pairs (sentence as list of words)
    vocab = set()
    with open("data/amazon_cells_labelled.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            message, label = row
            message = word_tokenize(message)
            message = [word.lower() for word in message if word.isalnum()]
            data.append((message, int(label)))
            vocab.update(message)

    vocab = list(dict.fromkeys(vocab))
    vocab = sorted(list(vocab))  # Contains all words from the file as single words and without duplicates
    all_labels = {
        0: 'negative',
        1: 'positive',
        2: 'greeting',
        # 3: 'goodbye',
        # 4: 'thanks'
    }

    x_train = []
    y_train = []
    for (tokenized_sentence, label) in data:
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
    hidden_size = 100
    output_size = len(all_labels)
    learning_rate = 0.001
    epochs = 1000

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

        print(
            f'\rEpoch: [{epoch + 1}/{epochs}] - Loss: {loss}',
            end=''
        )

    save_data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "vocab": vocab,
        "all_labels": all_labels
    }

    FILE = "chatbot_ann.pth"
    torch.save(save_data, FILE)


def run():
    print("Type [bot] to start the chatbot or [train] to train the network:")
    while True:
        user_input = input()
        if user_input.lower() == "train":
            train()
            return
        elif user_input.lower() == "bot":
            break

    bot = chatbot.Chatbot()
    bot.print_response()
    while True:
        bot.handle_input()


if __name__ == '__main__':
    run()
