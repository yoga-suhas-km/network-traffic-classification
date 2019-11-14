import os
import glob
import pandas as pd
import numpy as np
from functools import partial
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LABELS = {}
counter = iter(range(20))

def pad_and_convert(s):
    """Collect 1000 bytes from packet payload. If payload length is less than
    1000 bytes, pad zeroes at the end. Then convert to integers and normalize."""
    """
    if len(s) < 2000:
        s += '00' * (2000-len(s))
    else:
        s = s[:2000]
    return [float(int(s[i]+s[i+1], 16)/255) for i in range(0, 2000, 2)]
    """
    if len(s) < 1000:
        s += '00' * (1000-len(s))
    else:
        s = s[:1000]
    return [float(int(s[i]+s[i+1], 16)/255) for i in range(0, 1000, 2)]
   
def read_file(f, label):
    df = pd.read_csv(f, index_col=None, header=0)
    df.columns = ['data']
    df['label'] = label
    return df

def preprocess(path):
    files = glob.glob(os.path.join(path, '*.txt'))
    list_ = []
    for f in files:
        label = f.split('/')[-1].split('.')[0]
        LABELS[label] = next(counter)
        labelled_df = partial(read_file, label=LABELS[label])
        list_.append(labelled_df(f))
    df = pd.concat(list_, ignore_index=True)
    return df

#def build_model(dropout_rate=0.01, optimizer='adam'):
def build_model(X_train, y_train,X_test, y_test):
    activation = 'relu'
    num_classes = len(LABELS)
    model = Sequential()
    model.add(Conv1D(16, strides=1, input_shape=(500, 1), activation=activation, kernel_size=3, padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, strides=1, activation=activation, kernel_size=3, padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv1D(64, strides=1, activation=activation, kernel_size=3, padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv1D(128, strides=1, activation=activation, kernel_size=3, padding='valid'))
    model.add(BatchNormalization())
    model.add(LSTM(100,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200,return_sequences=True))
    model.add(Dropout(0.4))
    model.add(Dense(200, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(108, activation=activation))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, verbose=1, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    return model

def main():
    df = preprocess(path='Dataset')
    df['data'] = df['data'].apply(pad_and_convert)
    X_train, X_test, y_train, y_test = train_test_split(df['data'], df['label'],
                                                        test_size=0.3, random_state=4)
    X_train = X_train.apply(pd.Series)
    X_test = X_test.apply(pd.Series)
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    #num_classes = len(LABELS)
    #y_train = to_categorical(y_train, num_classes)
    #y_test = to_categorical(y_test, num_classes)
    #print(y_train.shape)
    
    model = build_model(X_train, y_train,X_test, y_test)
    #model.fit(X_train, y_train, verbose=1, epochs=10, batch_size=32, validation_data=(X_test, y_test))


    # predict crisp classes for test set
    y_pred = model.predict_classes(X_test)#, verbose=0)
    #print(yhat_probs.shape)
    #print(y_pred.shape)


    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred))
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_pred,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred))
    print('F1 score: %f' % f1)
    

if __name__ == '__main__':
    main()
