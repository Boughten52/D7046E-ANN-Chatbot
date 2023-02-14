import chatbot
import dataloader


def run():
    # Code below was just used for debugging the dataloader
    # loader = dataloader.DataLoader("dataloader/amazon_cells_labelled.txt")
    # print(loader.data)
    # print(loader.vocab_size)

    bot = chatbot.Chatbot()
    bot.print_response()
    while True:
        bot.handle_input()


if __name__ == '__main__':
    run()
