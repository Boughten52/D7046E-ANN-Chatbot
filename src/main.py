import chatbot
import dataloader
import vocabulary
import embedder


def debug():
    # DEBUG THE AMAZON DATALOADER
    # loader = dataloader.AmazonDataLoader("dataloader/amazon_cells_labelled.txt")
    # print(loader.vocabulary)
    # print(loader.vocab_size)
    # print(loader.data["Sentence"])

    # DEBUG THE AMAZON VOCABULARY
    amazon_vocabulary = vocabulary.AmazonVocabulary()
    # print(amazon_vocabulary.token2index)

    # DEBUG THE BOW EMBEDDER
    sentence = "Hello, this product is quite nice and it is also very good!"
    embedded_sentence = embedder.bow_embedder(sentence, amazon_vocabulary)
    # print(sum(embedded_sentence))


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
