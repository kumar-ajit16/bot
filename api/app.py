import os
import json
import pickle
import random
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from keras.models import load_model
import pyjokes
import requests



def extract_name(text: str):
    match = re.search(r"\b(i am|i'm|my name is)\s+([a-zA-Z]+)", text.lower())
    if match:
        return match.group(2).capitalize()
    return None

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

def has_known_words(sentence: str):
    tokens = tokenize(sentence)
    return any(word in words for word in tokens)


def predict_class(sentence: str):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]

    ERROR_THRESHOLD = 0.7

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

def ask_rasa(message: str):
    try:
        res = requests.post(
            "http://localhost:5005/webhooks/rest/webhook",
            json={"sender": "user", "message": message},
            timeout=1.5  
        )
        if res.ok and res.json():
            return res.json()[0]["text"]
    except Exception:
        pass
    return None



@app.post("/chat")
def chat(req: ChatRequest):
    user_message = req.message.strip().lower()
    tokens = tokenize(user_message)

    SMALL_TALK_PHRASES = [
    "how are you",
    "how are you doing",
    "what is rasa",
    "tell me about rasa",
    "explain rasa",
    "what is the time",
    "what is time",
    "current time",
    "current date",
    "date and time",
    "bye",
    "goodbye"
]

    if any(p in user_message for p in SMALL_TALK_PHRASES):
        rasa_reply = ask_rasa(user_message)
        if rasa_reply:
            return {"reply": rasa_reply}

    if any(w in tokens for w in ["joke", "jokes", "funny", "laugh", "bored"]):
        return {"reply": pyjokes.get_joke(category="neutral", language="en")}

    if len(tokens) == 1:
        if tokens[0] == "python":
            return {"reply": "What do you want to know about Python?"}
        if tokens[0] == "django":
            return {"reply": "What do you want to know about Django?"}

    name = extract_name(user_message)
    tag = predict_class(user_message)

    if not tag:
        return {"reply": "Sorry, I didn’t understand that "}

    reply = get_response(tag)


    if name and tag == "greeting":
        reply = f"Hi {name}  How can I help you today?"

    return {"reply": reply}

