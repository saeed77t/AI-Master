#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


# Read the Excel file into a pandas DataFrame
df = pd.read_csv('Dataset/Text_Emotion_Data.csv')
pattern = r'[^a-zA-Z]'

# Define a function to tokenize text into word sequences and remove stopwords
def tokenize_text(text):
    # Remove non-letter characters using the defined pattern
    text = re.sub(pattern, ' ', text)
    # Convert the text to lowercase
    text = text.lower()
    # Tokenize the text into word sequences
    word_sequences = text.split()
    # Remove stopwords using the provided list
    with open('Dataset/stopwords.txt', 'r') as f:
        stopwords = f.read().splitlines()
    # Remove words with length less than or equal to 2
    word_sequences = [word for word in word_sequences if (word not in stopwords and len(word) > 2)]
    return word_sequences

# Tokenize each row of the text column into word sequences using the defined function
df['word_sequences'] = df['Text'].apply(tokenize_text)

# Find the maximum length of a word sequence
max_len = max(df['word_sequences'].apply(len))

# Define a function to pad the sequences to the maximum length
def pad_sequence(sequence):
    padded_sequence = sequence[:max_len] + ['']*(max_len-len(sequence))
    return padded_sequence

# Pad each sequence to the maximum length
df['word_sequences'] = df['word_sequences'].apply(pad_sequence)


# In[ ]:





# In[3]:


classes = np.unique(df['Label'])
print('labels = ',classes)

label = np.array(df['Label'])


# In[4]:


## decpricated , doent keep the order of the code ! !!!!
# # Combine all word sequences into a single list
# all_sequences = []
# for seq in df['word_sequences']:
#     for word in seq:
#         all_sequences.append(word)

# # Create a dictionary with unique words as keys and their corresponding index as values
# word_dict = {}
# index = 0
# for word in set(all_sequences):
#     word_dict[word] = index
#     index += 1

# # Convert each word sequence into a one-hot encoded numerical vector
# num_vectors = []
# for sequence in df['word_sequences']:
#     vector = [0] * len(word_dict)
#     for word in sequence:
#         if word in word_dict:
#             index = word_dict[word]
#             vector[index] = 1
#     num_vectors.append(vector)

# # Convert the list of numerical vectors into a numpy array
# X = np.array(num_vectors)

# # Print the shape of the resulting numerical vectors
# print('Shape of numerical vectors:', X.shape)


# In[5]:


# new mwthod :
# Combine all word sequences into a single list
all_sequences = []
for seq in df['word_sequences']:
    all_sequences.append(seq)

# Create a dictionary with unique words as keys and their corresponding index as values
word_dict = {}
index = 0
for seq in all_sequences:
    for word in seq:
        if word not in word_dict:
            word_dict[word] = index
            index += 1

# Convert each word sequence into a numerical vector with the corresponding index in the dictionary
num_vectors = []
for sequence in all_sequences:
    vector = []
    for word in sequence:
        if word in word_dict:
            index = word_dict[word]
            vector.append(index)
    num_vectors.append(vector)

# Convert the list of numerical vectors into a numpy array
X = np.array(num_vectors)

# Print the shape of the resulting numerical vectors
print('Shape of numerical vectors:', X.shape)


# In[6]:


X


# In[7]:


y = label
# Split last 150 text of each class for the test dataset
test_data = []
for c in classes:
    class_data = [(X[i], y[i]) for i in range(len(X)) if y[i] == c]
    test_data.extend(class_data[-150:])

# Use the rest of the data for training
train_data = []
for i in range(len(X)):
    found = False
    for j in range(len(test_data)):
        if all(X[i] == test_data[j][0]) and y[i] == test_data[j][1]:
            found = True
            break
    if not found:
        train_data.append((X[i], y[i]))

# Separate the input features and labels for the training and test sets
X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Build the model architecture
model = Sequential()
model.add(SimpleRNN(units=64, input_shape=(max_len, len(word_dict)), activation='tanh'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=len(classes), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32)

# Evaluate the model on the train dataset
train_loss, train_acc = model.evaluate(X_train, to_categorical(y_train), verbose=0)
print('Train Accuracy:', train_acc)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test), verbose=0)
print('Test Accuracy:', test_acc)


# In[ ]:




