import os
import json
import pickle
import random
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

with open(os.path.join(MODEL_DIR, "intents.json"), "r") as f:
    intents = json.load(f)

with open(os.path.join(MODEL_DIR, "words.pkl"), "rb") as f:
    words = pickle.load(f)

with open(os.path.join(MODEL_DIR, "classes.pkl"), "rb") as f:
    classes = pickle.load(f)

model = load_model(os.path.join(MODEL_DIR, "chatbot_ajit.h5"))

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())

def bag_of_words(sentence: str):
    tokens = tokenize(sentence)
    return np.array([1 if w in tokens else 0 for w in words])

def predict_class(sentence: str):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [(classes[i], r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0][0] if results else None

def get_response(tag: str):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Can you please rephrase?"

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    tag = predict_class(req.message)
    reply = get_response(tag) if tag else "Can you please rephrase?"
    return {"reply": reply}
