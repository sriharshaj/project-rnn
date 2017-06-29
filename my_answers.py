import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    uniq_chars = set(text)
    import string
    non_english_chars = list(uniq_chars - set(string.ascii_lowercase) - {'.', ',','?','"',' ', "'", '!'})
    for char in non_english_chars:
        text = text.replace(char, ' ')

    text = text.replace('  ',' ')
    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    inputs = []
    outputs = []
    index = 0
    text_length = len(text)
    while index + window_size < text_length:
        outputs.append(text[index+window_size])
        inputs.append(text[index:index+window_size])
        index += step_size
    return inputs,outputs
