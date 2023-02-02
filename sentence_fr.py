import csv
import io
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import LSTM, Embedding, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

reuse = True
maxData = 64
max_num_words=2000
batch_size = 32
epochs = 1
latent_dim=512
model_name = "model_sentence_fr"
data_folder = "fr/"
clips_folder = os.path.join(data_folder, "clips")
block_length = 0.500#->500ms
frame_length=512
voice_max_length = int(8/block_length)#->8s
print("voice_max_length:", voice_max_length)

def audioToTensor(filepath:str):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    audio_length = int(audioSR * block_length)#20-> 50ms 40 -> 25ms
    frame_step = int(audioSR * 0.010)# 128 when rate is 1600 -> 8ms

    required_length = audio_length*voice_max_length    
    if len(audio)<required_length:
        audio = tf.concat([np.zeros([required_length-len(audio)]), audio], 0)
    else:
        audio = audio[-required_length:]

    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spectrogram = (tf.math.log(tf.abs(tf.math.real(spectrogram)))/tf.math.log(tf.constant(10, dtype=tf.float32))*20)-60
    spectrogram = tf.where(tf.math.is_nan(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    spectrogram = tf.where(tf.math.is_inf(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    return spectrogram

def sampleFromFile(filepath):
    print("Load data from", filepath)
    with open(filepath) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      next(reader)#skip header
      for row in reader:
        sentence = row[2].replace(".", "")
        wordList = ("start " + sentence + " end").split(" ")
        if(len(wordList)<5):
            continue
        return row[1]+".wav"

samplePath = sampleFromFile(os.path.join(data_folder, 'train.tsv'))
testParts = audioToTensor(os.path.join(clips_folder, samplePath))print("testParts", testParts.shape)

def loadDataFromFile(filepath):
    print("Load data from", filepath)
    dataVoice, dataString = [], []
    string_max_length = 0
    with open(filepath) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      next(reader)#skip header
      for row in reader:
        if len(dataString)>maxData:
            break
        sentence = row[2].replace(".", "")
        wordList = ("start " + sentence + " end").split(" ")
        if(len(wordList)<5):
            continue
        print(row[1], row[2], wordList)
        string_max_length = max(len(wordList), string_max_length)
        dataString.append(wordList)
        #dataVoice.append(row[1].replace(".mp3", '.wav'))
        dataVoice.append(row[1]+'.wav')
    return dataVoice, dataString, string_max_length

dataVoice, dataString, string_max_length = loadDataFromFile(os.path.join(data_folder, 'train.tsv'))

print("voice_max_length: ", voice_max_length)
print("string_max_length: ", string_max_length)
tokenizer = Tokenizer(num_words=max_num_words, lower=True, oov_token="<rare>")
tokenizer.fit_on_texts(dataString)
with io.open('tokenizer.txt', 'w', encoding='utf-8') as f:
    for word, index in tokenizer.word_index.items():
        f.write(word + ":" + str(index) + "\n")
vocab_size = min(len(tokenizer.word_index) + 1, max_num_words)
print('Vocabulary Size: %d' % vocab_size)

def prepareData(dataString, dataVoice):
    X_voice, X_string, Y_string = list(), list(), list()
    all_seq = tokenizer.texts_to_sequences(dataString)
    for i, seq in enumerate(all_seq):
        voice =  dataVoice[i]
        for j in range(1, len(seq)):
            in_seq, out_seq = seq[:j], [seq[j]]
            in_seq = pad_sequences([in_seq], maxlen=string_max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X_voice.append(voice)
            X_string.append(in_seq)
            Y_string.append(out_seq)
    return X_voice, X_string, Y_string

X_voice, X_string, Y_string = prepareData(dataString, dataVoice)
print("len(X_voice): ", len(X_voice))

class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_voice, x_string, y_string, batch_size):
        self.x_voice, self.x_string, self.y_string = x_voice, x_string, y_string
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.x_voice) // self.batch_size

    def __getitem__(self, idx):
        batch_x_string = self.x_string[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_string = self.y_string[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_voice = np.zeros((self.batch_size, testParts.shape[0], testParts.shape[1]))
        for i in range(0, batch_size):
            batch_x_voice[i] = audioToTensor(os.path.join(clips_folder, self.x_voice[idx * self.batch_size + i]))
        batch_x_string = np.array(batch_x_string)
        batch_y_string = np.array(batch_y_string)
        return [batch_x_voice, batch_x_string], batch_y_string

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

if os.path.exists(model_name) and reuse:
    print("Load: " + model_name)
    model = load_model(model_name)
else:
    encoder_inputs = Input(shape=(testParts.shape[0], testParts.shape[1]))
    encoder_inputs = tf.expand_dims(encoder_inputs, axis=-1)
    
    preprocessing = preprocessing.Resizing(400, testParts.shape[1]//2)(encoder_inputs)
    normalization = BatchNormalization()(preprocessing)

    split = tf.keras.layers.Reshape((voice_max_length, -1, normalization.shape[2], normalization.shape[3]))(normalization)

    conv2d = TimeDistributed(Conv2D(34, 3, activation='relu'))(split)
    conv2d = TimeDistributed(Conv2D(64, 3, activation='relu'))(conv2d)
    maxpool = TimeDistributed(MaxPooling2D())(conv2d)
    dropout = TimeDistributed(Dropout(0.25))(maxpool)
    flatten = TimeDistributed(Flatten())(dropout)

    encoder_lstm = LSTM(units=latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(flatten)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(string_max_length))
    dec_emb = Embedding(vocab_size, latent_dim)(decoder_inputs)
    decoder_outputs = LSTM(units=latent_dim)(dec_emb, initial_state=encoder_states)
    decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    tf.keras.utils.plot_model(model, to_file='model_sentence.png', show_shapes=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(MySequence(X_voice, X_string, Y_string, batch_size), epochs=epochs, batch_size=batch_size)
model.save(model_name)

def decode_sequence(input_seq):
    decoded_sentence = tokenizer.texts_to_sequences(["start"])[0]
    while len(decoded_sentence) < string_max_length:
        sequence = pad_sequences([decoded_sentence], maxlen=string_max_length)
        output_tokens = model.predict([input_seq, sequence], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0])
        decoded_sentence.append(sampled_token_index)
    return tokenizer.sequences_to_texts([decoded_sentence])[0]

print("Test voice recognition")
for i in range(0, 5):
    test_path = dataVoice[i]
    test_string = dataString[i]
    print("test_string: ", test_string)
    test_voice = audioToTensor(os.path.join(clips_folder, test_path))
    print(np.array([test_voice]).shape)
    decoded_sentence = decode_sequence(np.array([test_voice]))
    print("decoded_sentence: ", decoded_sentence)
