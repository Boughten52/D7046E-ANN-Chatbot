import copy

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import chatbot
import embedding
import ann
import model_tester


def validate(valid_loader, model, criterion, device):
    losses = []
    for (words, labels) in valid_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.int64)
        outputs = model(words)
        losses.append(criterion(outputs, labels))
    return sum(losses) / len(losses)


def train(dataset_type, use_validation):
    """Trains the ANN on the Amazon reviews dataset (expanded with greetings, goodbyes and thanks)"""

    if dataset_type == "coursera":
        file_path_train_set = "data/coursera_pre_strat_train_1800.csv"
        all_labels = {0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}
    else:
        file_path_train_set = "data/amazon_cells_labelled_train.txt"
        all_labels = {0: 'negative', 1: 'positive', 2: 'greeting', 3: 'goodbye', 4: 'thanks'}

    # Dataloader hyperparameters
    batch_size = 50
    shuffle = True

    # Datasets
    x_train, y_train, vocab = embedding.read_file_to_tensor_and_vocab(file_path_train_set)
    dataset = embedding.ChatBotDataset(x_train, y_train)

    # Dataloaders
    if use_validation:
        train_loader_size = int(len(dataset) * 0.9)
        valid_loader_size = len(dataset) - train_loader_size
        dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [train_loader_size, valid_loader_size])

        train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=shuffle)
    else:
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        valid_loader = None

    # Network hyperparameters
    input_size = len(vocab)
    hidden_size = 200
    output_size = len(all_labels)
    learning_rate = 0.01
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the network
    model = ann.ChatBotANN(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

    # Train the network
    best_network = copy.deepcopy(model)
    validation_losses = [numpy.Infinity]
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

        # Validation
        if use_validation:
            validation_loss = validate(valid_loader, model, criterion, device)
            if validation_loss < min(validation_losses):
                best_network = copy.deepcopy(model)
                print(f'Best network updated with validation loss: {validation_loss}')
            validation_losses.append(validation_loss)
        else:
            best_network = copy.deepcopy(model)

        print(f'Epoch: [{epoch + 1}/{epochs}] - Loss: {loss}')

    # Save the model to file
    save_data = {
        "model_state": best_network.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "vocab": vocab,
        "all_labels": all_labels
    }

    if dataset_type == "coursera":
        FILE = "chatbot_ann_coursera.pth"
    else:
        FILE = "chatbot_ann.pth"

    torch.save(save_data, FILE)


def run():
    """First asks the user if it wants to train/test the network or chat, then starts the chat-loop"""

    print("Type [bot] to start the chatbot or [train / test] to train/test the network:")
    while True:
        user_input = input()
        if user_input.lower() == "train":
            print("Type [validation] to use validation during training. For no validation, press [ENTER].")
            if input() == "validation":
                use_validation = True
            else:
                use_validation = False

            print("Type [coursera] to train on the Coursera dataset. Otherwise press [ENTER] for Amazon default.")
            if input() == "coursera":
                train(dataset_type="coursera", use_validation=use_validation)
            else:
                train(dataset_type="amazon", use_validation=use_validation)
            return

        elif user_input.lower() == "test":
            print("Type [coursera] to test on the Coursera dataset. Otherwise press [ENTER] for Amazon default.")
            if input() == "coursera":
                tester = model_tester.Tester(dataset_type="coursera")
            else:
                tester = model_tester.Tester(dataset_type="amazon")
            tester.run_test()
            return

        elif user_input.lower() == "bot":
            print("Type [coursera] to chat with the Coursera bot. Otherwise press [ENTER] for Amazon default.")
            if input() == "coursera":
                bot = chatbot.Chatbot(dataset_type="coursera")
            else:
                bot = chatbot.Chatbot(dataset_type="amazon")
            break

    bot.print_response()
    while True:
        bot.handle_input()


if __name__ == '__main__':
    run()
