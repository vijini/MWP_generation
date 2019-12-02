from tensorflow import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
# load ascii text and covert to lowercase

filename = "dataset_new_new.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
'''
# define the checkpoint
#filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-forPOSWithNewNew.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=15, batch_size=128, callbacks=callbacks_list)
'''
# load the network weights
filename = "weights-improvement-15-0.5327-forPOSWithNewNew.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))

import sys
import numpy
out = open('output.txt', 'w')
# pick a random seed
for j in range(1):
        
  start = numpy.random.randint(0, len(dataX)-1)
  pattern = dataX[start]
  sentence= ""
  print ("\n")
  #print ("\nSeed:")
  seed =''
  #print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
  seed = ''.join([int_to_char[value] for value in pattern])
  sentence += seed
  #print(seed)
  # generate characters
  #print("\nGenerated:")
  
  for i in range(125):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    sentence += result
    seq_in = [int_to_char[value] for value in pattern]
    
    out.write(str(result))
    
    sys.stdout.write(result)
    
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
  #print(sentence)
  #output_filename = 'output.txt'
  #save_doc(result, output_filename)
out.close()  
print( "\nDone.")
print(sentence)
'''
text = word_tokenize(sentence)
nltk.pos_tag(text)
'''
#applying POS tags to the genretaed sentence
tagged_sentences = []
sentences = sentence.decode('utf-8')
tagged = nltk.sent_tokenize(sentences.strip())
tagged = [nltk.word_tokenize(sent) for sent in tagged]          
tagged = [nltk.pos_tag(sent) for sent in tagged]
tagged_sentences.append(tagged[0])
print(tagged[0])

#separating the words from the tags    
sentences, sentence_tags =[], []  

for tagged_sentence in tagged_sentences:
    sentence, tags = zip(*tagged_sentence)
    sentences.append(np.array(sentence))
    sentence_tags.append(np.array(tags))

# Let's see how a sequence looks
 
print(sentences[0])
print(sentence_tags[0])

for x in sentences[0]:
        if(x=='less'):
                for y in sentence_tags[0]:
                        indices = [i for i, y in enumerate(sentences) if y == "CD"]
                        
                        
        



