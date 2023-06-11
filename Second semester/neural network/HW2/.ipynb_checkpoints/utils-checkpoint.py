import pandas as pd
import numpy as np
import re

class TextPreprocessor:
    def __init__(self, data_file='Dataset/Text_Emotion_Data.csv', stopwords_file='Dataset/stopwords.txt'):
        self.data_file = data_file
        self.stopwords_file = stopwords_file
        self.df = None  # DataFrame to store the data
        self.max_len = None  # Maximum length of a word sequence
        self.word_dict = None  # Dictionary to map words to their corresponding indices
        self.X = None  # Numpy array to store the converted numerical vectors

    def read_data(self):
        # Read the data file into a DataFrame
        self.df = pd.read_csv(self.data_file)

    def tokenize_text(self, text):
        pattern = r'[^a-zA-Z]'
        text = re.sub(pattern, ' ', text)
        text = text.lower()
        word_sequences = text.split()
        with open(self.stopwords_file, 'r') as f:
            stopwords = f.read().splitlines()
        word_sequences = [word for word in word_sequences if (word not in stopwords and len(word) > 2)]
        return word_sequences

    def preprocess_text(self):
        # Tokenize text in each row of the 'Text' column into word sequences
        self.df['word_sequences'] = self.df['Text'].apply(self.tokenize_text)

        # Find the maximum length of a word sequence
        self.max_len = max(self.df['word_sequences'].apply(len))

        def pad_sequence(sequence):
            # Pad the sequence to the maximum length
            padded_sequence = sequence[:self.max_len] + [''] * (self.max_len - len(sequence))
            return padded_sequence

        # Pad each sequence to the maximum length
        self.df['word_sequences'] = self.df['word_sequences'].apply(pad_sequence)

        all_sequences = []
        for seq in self.df['word_sequences']:
            all_sequences.append(seq)

        self.word_dict = {}
        index = 0
        for seq in all_sequences:
            for word in seq:
                if word not in self.word_dict:
                    # Create a dictionary with unique words as keys and their corresponding index as values
                    self.word_dict[word] = index
                    index += 1

        num_vectors = []
        for sequence in all_sequences:
            vector = []
            for word in sequence:
                if word in self.word_dict:
                    # Convert each word sequence into a numerical vector with the corresponding index in the dictionary
                    index = self.word_dict[word]
                    vector.append(index)
            num_vectors.append(vector)

        # Convert the list of numerical vectors into a numpy array
        self.X = np.array(num_vectors)
        
        
        

def RemoveStops(text, stop_words):
    # Remove special characters and lowercase the text
    text = re.sub(r"[^\w\s]", "", text.lower())
    # Remove stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text





def create_word_dictionary(dataset):
    
    #old 
    # # Get all the unique words from your dataset
    # unique_words = []
    # for sequence in df['Padded_Text']:
    #     unique_words.extend(sequence)

    # # Remove duplicates and sort the unique words
    # unique_words = sorted(set(map(str, unique_words)))

    # # Create a dictionary of unique words
    # word_dictionary = {word: index for index, word in enumerate(unique_words)}

    # # Print the number of unique words
    # print("Number of Unique Words:", len(unique_words))

    # # Print the word dictionary
    # print("Word Dictionary:")
    # print(len(word_dictionary))


    
    
    # Get all the unique words from the dataset
    unique_words = []
    for sequence in dataset:
        unique_words.extend(sequence)

    # Remove duplicates and sort the unique words
    unique_words = sorted(set(map(str, unique_words)))

    # Create a dictionary of unique words
    word_dictionary = {word: index for index, word in enumerate(unique_words)}

    return word_dictionary






def split_train_test_data(X, labels, test_size=150):
    #old :
        # X = np.array(df['Padded_Text'].tolist())
    # # Define the labels
    # labels = df['Label'].values
    # classes = np.unique(labels)
    # label_map = {label: i for i, label in enumerate(classes)}
    # y = np.array([label_map[label] for label in labels])

    # # Split last 150 text of each class for the test dataset
    # test_data = []
    # for c in classes:
    #     class_data = [(X[i], y[i]) for i in range(len(X)) if y[i] == label_map[c]]
    #     test_data.extend(class_data[-150:])

    # # Use the rest of the data for training
    # train_data = []
    # for i in range(len(X)):
    #     found = False
    #     for j in range(len(test_data)):
    #         if all(X[i] == test_data[j][0]) and y[i] == test_data[j][1]:
    #             found = True
    #             break
    #     if not found:
    #         train_data.append((X[i], y[i]))

    # # Separate the input features and labels for the training and test sets
    # X_train, y_train = zip(*train_data)
    # X_test, y_test = zip(*test_data)


    # X_train = np.array(X_train)
    # y_train = np.array(y_train)

    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
    
    
    
    # Define the labels
    classes = np.unique(labels)
    label_map = {label: i for i, label in enumerate(classes)}
    y = np.array([label_map[label] for label in labels])

    # Split last test_size texts of each class for the test dataset
    test_data = []
    for c in classes:
        class_data = [(X[i], y[i]) for i in range(len(X)) if y[i] == label_map[c]]
        test_data.extend(class_data[-test_size:])

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

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test




def ThreeDim_one_hot_encoder(dataset, dictionary):
    max_length = dataset.shape[1]
    vocab_size = max(dictionary.values()) + 1

    one_hot_word_array = np.zeros((dataset.shape[0], max_length, vocab_size))

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            word = dataset[i, j]
            if word != '':
                if word in dictionary:
                    index = dictionary[word]
                    one_hot_word_array[i, j, index] = 1
                else:
                    # Handle unknown words (not present in the dictionary)
                    one_hot_word_array[i, j, 0] = 1

    return one_hot_word_array


