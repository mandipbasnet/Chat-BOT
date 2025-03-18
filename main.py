import tensorflow as tf
import numpy as np
import random
import json
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Data preparation
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = pattern.split()
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([w.lower() for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
x_train = np.array(list(training[:, 0]))
y_train = np.array(list(training[:, 1]))

# Model
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

# Prediction function
def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    return classes[np.argmax(res)]

# Chat
while True:
    message = input('You: ')
    if message.lower() == 'quit':
        break
    tag = predict_class(message)
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            print('Bot:', response)

print('Chatbot is ready! Type "quit" to exit.')
