import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, GlobalMaxPooling1D, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load Dataset
data = pd.read_csv('data/syns.csv')
x_train = data["Data"]
y_train = data["Label"]


VOCAB_LENGTH = 20000
tokenizer = Tokenizer(num_words=VOCAB_LENGTH)
tokenizer.fit_on_texts(x_train)
sequence_train = tokenizer.texts_to_sequences(x_train)
#sequence_test = tokenizer.texts_to_sequences(x_test)

V = len(tokenizer.word_index)

x_train = pad_sequences(sequence_train)
T = x_train.shape[1]
#x_test = pad_sequences(sequence_test, maxlen=T)


D = 20
M = 15
K = 1
i = Input(shape=(T,))
x = Embedding(V + 1, D)(i)
x = Conv1D(32, 3, padding='same', activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, padding='same', activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, padding='same', activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.compile(loss='binary_crossentropy', metrics=[
              'accuracy'], optimizer='adam')

r = model.fit(x_train, y_train, epochs=10)

plt.plot(r.history["loss"], label="loss")
#plt.plot(r.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plt.plot(r.history["accuracy"], label="accuracy")
#plt.plot(r.history["val_accuracy"], label="val_laccuracy")
plt.legend()
plt.show()

model.save('models/text_calssification_cnn.h5')

print(model.summary())
