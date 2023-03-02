import random
import torch
import json
from nltk.tokenize import word_tokenize
from sys import exit

from src.ann import ChatBotANN
from src.embedding import bow_embedder


class Chatbot:
    """Class that handles input/output by picking a response for a user input"""

    def __init__(self):
        self.exit_conditions = ["exit", "quit"]
        self.user_input = ""
        self.response = "Hi, I'm a Chatbot. Pleased to meet you! Type [exit] or [quit] to stop talking to me."
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('data/responses.json', 'r') as f:
            self.classes = json.load(f)

        FILE = "chatbot_ann.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.vocab = data["vocab"]
        self.all_labels = data["all_labels"]
        model_state = data["model_state"]

        self.model = ChatBotANN(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def print_response(self):
        print(self.response)

    def handle_input(self):
        self.user_input = input()
        if self.user_input in self.exit_conditions:
            self.response = "The bot has exited"
            self.print_response()
            exit()

        self.user_input = word_tokenize(self.user_input)
        x = bow_embedder(self.user_input, self.vocab)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x)

        output = self.model(x)
        _, predicted = torch.max(output, dim=1)
        label = self.all_labels[predicted.item()]

        probabilities = torch.softmax(output, dim=1)
        probability = probabilities[0][predicted.item()]

        if probability.item() > 0.7:
            for sentence_class in self.classes["classes"]:
                if label == sentence_class["label"]:
                    self.response = random.choice(sentence_class["responses"])
        else:
            self.response = "Unfortunately I don't understand what you're trying to say :("

        self.print_response()
