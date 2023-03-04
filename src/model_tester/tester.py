import csv
import torch
import json
from nltk.tokenize import word_tokenize
from sys import exit

from src.ann import ChatBotANN
from src.embedding import bow_embedder

class tester:
    """Class that handles input/output by picking a response for a user input"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('data/responses_coursera.json', 'r') as f:
            self.classes = json.load(f)

        FILE = "chatbot_ann_coursera.pth"
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


    def result(self):
        print("Result : _______")

    def run_test(self):
        with open("data/coursera_pre_valid_2500_tab_0-4.csv", "r") as t:
            reader = csv.reader(t, delimiter="\t")
            for row in reader:
                message, label_true = row

                x = bow_embedder(message, self.vocab)
                x = x.reshape(1, x.shape[0])
                x = torch.from_numpy(x)

                output = self.model(x)

                _, predicted = torch.max(output, dim=1)

                print(message)
                print(f'Ture Class: [{label_true}] / predicted : [{predicted.item()}]')


                # label = self.all_labels[predicted.item()]
                #
                # print(message)
                # print(label_true)
                # print(label)



