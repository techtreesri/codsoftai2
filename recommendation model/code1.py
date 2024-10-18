# Import necessary libraries
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model
from keras.layers import Input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import string
from pickle import dump
from pickle import load

# Load the VGG16 model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define the image captioning model
def define_model(vocab_size, max_length):
    # features from the CNN model compressed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # merge it [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    return model

# Create the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_list)

# Convert the text data into sequences
sequences = tokenizer.texts_to_sequences(desc_list)

# Pad the sequences
padded = pad_sequences(sequences, maxlen=max_length)

# One-hot encode the output
output = to_categorical(padded)

# Split the data into training and testing sets
trainX, trainY, testX, testY = train_test_split(padded, output, test_size=0.2, random_state=42)

# Create the data generator
def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            # retrieve photo features
            feature = features[key][0]
            inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield [[inp_image, inp_sequence], op_word]

# Train the model
model = define_model(vocab_size, max_length)
model.fit(data_generator(train_descriptions, train_features, tokenizer, max_length), epochs=10, steps_per_epoch=len(train_descriptions))

# Evaluate the model
loss = model.evaluate(data_generator(test_descriptions, test_features, tokenizer, max_length), steps=len(test_descriptions))
print('Test Loss: %f' % loss)

# Use the model to generate captions
def generate_caption(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'start'
    # iterate the caption generation process
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # stops if we cannot map the word
        if out_word == 'none':
            break
        # add to input
        in_text += ' ' + out_word
    return in_text

# Test the model
photo = load_photo('image.jpg')
description = generate_caption(model, tokenizer, photo, max_length)
print(description)
