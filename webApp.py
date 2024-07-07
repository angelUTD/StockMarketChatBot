import openai
from flask import Flask, render_template, request
import os
import fineTune

app = Flask(__name__)


#This class is used for the messages of the bot and user input.
class Message:
    def __init__(self, text, type):
        self.text = text
        self.type = type

#Stpre both the user and chat gpt's response so they are not removed when submitting another response.
messages = []

@app.route("/")
def index():
    return render_template("index.html", messages=messages)

@app.route("/", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = fineTune.tune(user_input)
    messages.append(Message(user_input, "user-message"))    #Stores the user's input in the array called messages
    messages.append(Message(response, "bot-message"))       #Stores the gtp's reponse in the array called messages
    return render_template("index.html", messages=messages) #Returns both to the website

if __name__ == "__main__":  #Starts the website at a specified port and ip address.
    port = int(os.environ.get("PORT", 5000))
    app.run(host = "127.0.0.1", port = port)


