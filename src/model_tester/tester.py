import csv
import torch
import json

from src.ann import ChatBotANN
from src.embedding import bow_embedder


class Tester:
    """Class that tests the network"""

    def __init__(self, dataset_type):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        match dataset_type:
            case "coursera":
                responses_path = "data/responses_coursera.json"
                network_path = "chatbot_ann_coursera.pth"
                self.test_set_path = "data/coursera_pre_strat_test_450.csv"
            case "amazon":
                responses_path = "data/responses.json"
                network_path = "chatbot_ann.pth"
                self.test_set_path = "data/amazon_cells_labelled_test.txt"
            case _:
                responses_path = ""
                network_path = ""
                self.test_set_path = ""

        with open(responses_path, 'r', encoding="UTF-8") as f:
            self.classes = json.load(f)

        FILE = network_path
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

    def run_test(self):
        correct = 0
        total = 0
        with open(self.test_set_path, "r", encoding="UTF-8") as t:
            reader = csv.reader(t, delimiter="\t")
            for row in reader:
                message, label_true = row

                x = bow_embedder(message, self.vocab)
                x = x.reshape(1, x.shape[0])
                x = torch.from_numpy(x)

                output = self.model(x)

                _, predicted = torch.max(output, dim=1)

                print(message)
                print(f'True Class: [{label_true}] / predicted : [{predicted.item()}]')
                total = total + 1
                if int(label_true) == predicted.item():
                    correct = correct + 1

        accuracy = correct / total
        print(f'Correct predictions: [{correct}]')
        print(f'Number of predictions: [{total}]')
        print(f'Accuracy: [{accuracy * 100}] %')
