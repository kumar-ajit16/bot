import random
import json
import pickle
import numpy as np
import tensorflow as tf
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "intents.json"), "r") as f:
    intents = json.load(f)


words = []
classes = []
documents = []

def tokenize(sentence):
    sentence = sentence.lower()
    return re.findall(r"\b\w+\b", sentence)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open(os.path.join(MODEL_DIR, "words.pkl"), "wb"))
pickle.dump(classes, open(os.path.join(MODEL_DIR, "classes.pkl"), "wb"))

training = []
output_empty = [0] * len(classes)

for word_list, tag in documents:
    bag = [1 if word in word_list else 0 for word in words]

    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(train_y[0]), activation="softmax"),
])

sgd = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9, nesterov=True
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]
)

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save(os.path.join(MODEL_DIR, "chatbot_ajit.h5"))
print("✅ Model created successfully")
