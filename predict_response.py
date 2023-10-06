#Text Data Preprocessing Lib
import nltk

import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

model=tensorflow.keras.models.load_model('./chatbot_model.h5')

intents=json.loads(open('./intents.json').read())
words=pickle.load(open('./words.pkl','rb'))
classes=pickle.load(open('./classes.pkl','rb'))

def preprocess_user_input(user_input):
    input_word_token_1=nltk.word_tokenize(user_input) # "how","are","you" "doing"
    input_word_token_2=get_stem_words(input_word_token_1)
    input_word_token_3=sorted(list(set(input_word_token_2))) #[]

    bag=[]
    bag_of_words=[]


