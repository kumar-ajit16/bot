import pyjokes
from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk import Tracker
from typing import Any, Text, List, Dict
from datetime import datetime

class ActionTellJoke(Action):

    def name(self) -> Text:
        return "action_tell_joke"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[Dict[Text, Any]]:

        joke = pyjokes.get_joke(category="neutral", language="en")
        dispatcher.utter_message(text=joke)
        return []

class ActionTellDateTime(Action):
    def name(self):
        return "action_tell_datetime"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        now = datetime.now()
        formatted = now.strftime("%A, %d %B %Y, %I:%M %p")

        dispatcher.utter_message(
            text=f"🕒 The current date and time is {formatted}"
        )
        return []