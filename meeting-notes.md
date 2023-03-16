# Meeting notes

### 2023-03-08
#### Participants: All
* Current progress:
  * Presentation slides are finished and the group is ready to present.
  * No further features have been implemented to the chatbot. All training and testing runs are completed.
* Next steps:
  * Present our project.

***

### 2023-03-06
#### Participants: All
* Note:
  * We skipped the meeting on 2023-03-02 and moved it to this date.
* Current progress:
  * A fully working chatbot trained on the Amazon reviews. It can also handle greetings, goodbyes and thanks.
  * Experimentation with larger datasets that has more labels (Coursera reviews with 0 - 5 stars).
* Next steps:
  * Evaluate the network and tune parameters to get better results.
  * Prepare for presentation on Thursday 2023-03-09.
* Next meeting:
  * Wednesday 2023-03-08 (9 AM CET).
  * Finish the presentation slides and practise presenting.

***

### 2023-02-27
#### Participants: All
* Current progress:
  * Rework of the provided dataloader code from Canvas.
  * The code now splits the data into training and validation vectors that are ready to be fed into the network.
* Next steps:
  * Train the network.
  * Implement saving and loading of the network.
  * Have the bot make predictions and choose from hardcoded responses.
* Next meeting:
  * 2023-03-02 during/after the scheduled project session.
  * Discuss how we can further improve the bot with more than binary classifications and further interactions.

***

### 2023-02-23
#### Participants: All
* Current progress:
  * Loading the Amazon reviews, creating a vocabulary from them and the BoW embedder are completed.
* Next steps:
  * We agreed to keep the project simple, so we know that we can manage it in the given timeframe.
  * We will wait with the Movie-Dialogs Corpus and stick to the Amazon reviews for now.
  * The idea is to train the model on the reviews to recognize if user input is positive, negative or somewhere in between.
  * Responses to a prediction will be hard coded.
  * We can later make the bot more interactive by responding to greetings, remembering the user's name etc.
  * Now we are ready to implement actual training of the network and create some responses for the bot to choose from.
* Next meeting:
  * 2023-02-27 (8:30 AM CET).
  * Hopefully the network will be done by then, and we can start working with user-bot interaction.

***

### 2023-02-20
#### Participants: All
* Current progress:
  * Most of us have now completed ANN exercise 2 which is useful for the project.
  * We presented some findings such as the PyTorch Chatbot Tutorial and the Cornell Movie-Dialogs Corpus.
  * Jonatan showed the project issues, project file structure and how to import packets in Python.
* Next steps:
  * Create a vocabulary for storing tokens and indices (Jonatan).
  * Implement word embedder (Georgios and Dimitra).
  * Start implementing and training the network (Folke and Jonatan).
  * We will use ANN exercise 2 and the PyTorch Chatbot Tutorial as help.
  * The network will be trained on the Amazon reviews and/or the Cornell Movie-Dialogs Corpus.
* Next meeting:
  * 2023-02-23 during or after the scheduled project session.
  * We will discuss our progress and how to proceed.

***

### 2023-02-13
#### Participants: All
* Familiarized ourselves with the project description.
* Discussed which IDE and source control to use; decided on PyCharm/VSCode and GitHub.
* Looked at the GitHub project board where issues can be created and assigned to a specific person.
* Next steps:
  * Download preferred IDE and get the test code running.
  * Create small, specific issues and start working with them (pick any issues you like).
  * Structure the project and start migrating code from the exercises.
* We meet during the scheduled project sessions on Thursdays, ready to present progress.
* Booked Mondays at 9 am (CET) for weekly meetings.
* Use the Discord server as a discussion forum for all kinds of questions.
