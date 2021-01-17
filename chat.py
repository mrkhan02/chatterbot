import random
import json
# Import the required module for text 
# to speech conversion 
from gtts import gTTS 

# This module is imported so that we can 
# play the converted audio 
import os 

# The text that you want to convert to audio 
import wikipedia


import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "colossus"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    xx=sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if(xx[:4] == "/pro"):
        result = wikipedia.summary(xx[5:], sentences = 2) 
        print(f"{bot_name}: {result}")
        myobj = gTTS(text=result, lang='en', slow=False)
        myobj.save("welcome.mp3")
        os.system("mpg321 welcome.mp3")
    elif prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                mytext = random.choice(intent['responses'])
                print(f"{bot_name}: {mytext}")
                myobj = gTTS(text=mytext, lang='en', slow=False) 
                myobj.save("welcome.mp3") 
                os.system("mpg321 welcome.mp3") 
    else:
        print(f"{bot_name}: I do not understand...")
        print('if your question is related to programming,')
        print("please type /pro question")
        print('example:')
        
        print("/pro Binary Search Tree")
        print('or visit https://kamandprompt.zulipchat.com/ ')
        mytext="I do not understand..."
        print("if your question is related to IIT Mandi,")
        print('Visit IIT Mandi wiki @ https://wiki.iitmandi.co.in/p/Main_Page ') 
        print("if your question is related to bot, visit our GitHub @ https://github.com/mrkhan02/chatterbot")