import chatbot
import dataloader


def debug():
    loader = dataloader.DataLoader("dataloader/amazon_cells_labelled.txt")
    print(loader.vocabulary)
    print(loader.vocab_size)


def run():
    print("Type [bot] to start the chatbot or [debug] to run the debug function:")
    while True:
        user_input = input()
        if user_input.lower() == "debug":
            debug()
            return
        elif user_input.lower() == "bot":
            break

    bot = chatbot.Chatbot()
    bot.print_response()
    while True:
        bot.handle_input()


if __name__ == '__main__':
    run()
