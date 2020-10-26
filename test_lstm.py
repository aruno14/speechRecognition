import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

maxData = 10
dataString = []
string_max_length = 0
with open('validated.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    #header = (client_id path sentence up_votes down_votes age gender accent locale segment)
    next(reader)#skip header
    for row in reader:
        if len(dataString) >= maxData:
            break
        sentence = ("start " + row[2] + " end").split(" ")
        dataString.append(sentence)
        string_max_length = max(len(sentence), string_max_length)

print("string_max_length: ", string_max_length)

tokenizer = Tokenizer(num_words=2000, lower=True, oov_token="<rare>")
tokenizer.fit_on_texts(dataString)
sequences = tokenizer.texts_to_sequences(dataString)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

X, y = list(), list()
for i, seq in enumerate(sequences):
    for j in range(1, len(seq)):
        in_seq, out_seq = seq[:j], seq[j]
        in_seq = pad_sequences([in_seq], maxlen=string_max_length)[0]
        in_seq = to_categorical([in_seq], num_classes=vocab_size)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X.append(in_seq)
        y.append(out_seq)
print('Total Sequences:', len(X))

model = Sequential()
model.add(Input(shape=(string_max_length, vocab_size)))
model.add(LSTM(units=32))
model.add(Dense(units=vocab_size, activation='softmax'))
model.summary()
tf.keras.utils.plot_model(model, to_file='model_lstm.png', show_shapes=True)

epoch = 300
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(np.array(X), np.array(y), epochs=epoch)
model.save_weights('model_lstm.h5')

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.savefig("learning-lstm.png")
plt.show()
plt.close()

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return Nones

input_str = "She is"
in_text = 'start ' +  input_str
print("in_text: ", in_text)
for i in range(string_max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=string_max_length)
    sequence = to_categorical([sequence], num_classes=vocab_size)[0]
    pred = model.predict(sequence, verbose=0)
    pred = np.argmax(pred)
    word = word_for_id(pred, tokenizer)
    if word is None:
        break
    in_text += ' ' + word
    if word == 'end':
        break
print("out_text: ", in_text)
