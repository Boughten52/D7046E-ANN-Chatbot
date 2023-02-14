class Chatbot:
    """Class that handles input/output by picking a response for a user input"""

    def __init__(self):
        self.exit_conditions = ["exit", "quit"]
        self.user_input = ""
        self.response = "Hi, I'm a Chatbot. Pleased to meet you! Type [exit] or [quit] to stop talking to me."

    def print_response(self):
        print(self.response)

    def handle_input(self):
        self.user_input = input()
        if self.user_input in self.exit_conditions:
            self.response = "Goodbye"
            self.print_response()
            exit()

        # TODO: Pick an appropriate response by predicting the output for the given input...
        self.response = "Generic response"
        self.print_response()
