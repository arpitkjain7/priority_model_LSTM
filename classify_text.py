import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, GlobalMaxPooling1D, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_text import remove_punctuations

# Load Dataset and reemove punctuations
data = pd.read_csv("data/orig_key_updated.csv")
df = remove_punctuations(data["Data"].values)
train = True
x_train, x_test, y_train, y_test = train_test_split(
    df, data["Label"], test_size=0.33)
print(x_train.shape)


VOCAB_LENGTH = 20000
tokenizer = Tokenizer(num_words=VOCAB_LENGTH, oov_token="UNK")
tokenizer.fit_on_texts(x_train)
sequence_train = tokenizer.texts_to_sequences(x_train)
# print(sequence_train)
sequence_test = tokenizer.texts_to_sequences(x_test)

word2idx = tokenizer.word_index
print(word2idx)
V = len(word2idx)
print(V)
#T = 10
if train:
    # padding
    x_train = pad_sequences(sequence_train)
    T = x_train.shape[1]
    print(f"max length is : {T}")
    x_test = pad_sequences(sequence_test, maxlen=T)
    # build a model
    M = 50
    K = 1
    D = 50
    i = Input(shape=(T,))
    x = Embedding(V + 1, D)(i)
    #x = Bidirectional(tf.keras.layers.LSTM(M))(x)
    x = LSTM(M, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(K, activation='sigmoid')(x)

    model = Model(i, x)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.01), metrics=["accuracy"])
    r = model.fit(x_train, y_train, validation_data=(
        x_test, y_test), epochs=200)

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()

    plt.plot(r.history["accuracy"], label="accuracy")
    plt.plot(r.history["val_accuracy"], label="val_laccuracy")
    plt.legend()
    plt.show()

    model.save('models/text_calssification_lstm.h5')
    print(model.summary())

else:
    model = tf.keras.models.load_model('models/text_calssification_lstm.h5')
    data = pd.read_csv("data/syns_test.csv")
    x_test = data["Test_data"].values
    sequence_test = tokenizer.texts_to_sequences(x_test)
    print(sequence_test)
    #x_test = pad_sequences(sequence_test, maxlen=T)
    print(sequence_test)
    r = model.predict(sequence_test[0])
    print(f"prediction prob is : {r}")
    print(f"max prob is : {max(r)}")
    class_pred = int(np.where(r == np.amax(r))[0])
    print(f"class predicted : {class_pred}")
