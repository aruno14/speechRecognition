import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import MeCab
import matplotlib.pyplot as plt

wakati = MeCab.Tagger("-Owakati -r /etc/mecabrc")

maxData = 1000
maxWord = 2000
dataString = []
string_max_length = 0
with open('validated.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    #header = (client_id path sentence up_votes down_votes age gender accent locale segment)
    next(reader)#skip header
    for row in reader:
        if len(dataString) >= maxData:
            break
        sentence = row[2].replace("。", "")
        wordList = wakati.parse("start " + sentence + " end").split()
        
        dataString.append(wordList)
        string_max_length = max(len(wordList), string_max_length)

print("string_max_length: ", string_max_length)

tokenizer = Tokenizer(num_words=maxWord, lower=True, oov_token="<rare>")
tokenizer.fit_on_texts(dataString)
sequences = tokenizer.texts_to_sequences(dataString)

vocab_size = min(len(tokenizer.word_index) + 1, maxWord)
print('Vocabulary Size: %d' % vocab_size)

X, Y = list(), list()
for i, seq in enumerate(sequences):
    for j in range(1, len(seq)):
        in_seq, out_seq = seq[:j], seq[j]
        in_seq = pad_sequences([in_seq], maxlen=string_max_length)[0]
        in_seq = to_categorical([in_seq], num_classes=vocab_size)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X.append(in_seq)
        Y.append(out_seq)
print('Total Sequences:', len(X))

model = Sequential()
model.add(Input(shape=(string_max_length, vocab_size)))
model.add(LSTM(units=32))
model.add(Dense(units=vocab_size, activation='softmax'))
model.summary()
tf.keras.utils.plot_model(model, to_file='model_lstm_ja.png', show_shapes=True)

epoch = 300
batch_size = 512
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(np.asarray(X), np.asarray(Y), epochs=epoch, batch_size=batch_size)
model.save_weights('model_lstm.h5')

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.savefig("learning-lstm_ja.png")
plt.show()
plt.close()

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return Nones

input_str = "彼女 は"
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
