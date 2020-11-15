import csv
import io
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import LSTM, Embedding, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow_io as tfio
import matplotlib.pyplot as plt

maxData = 30
model_name = "model_sentence_fr"
block_length = 0.050#->500ms
voice_max_length = int(10/block_length)#->2s
print("voice_max_length:", voice_max_length)
def audioToTensor(filepath):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    audio_length = int(audioSR * block_length)#20-> 50ms 40 -> 25ms
    frame_step = int(audioSR * 0.008)# 128 when rate is 1600 -> 8ms
    audio_clean = tf.constant([], tf.float32)
    audio_length_clean = audioSR//20#50ms
    for i in range(0, len(audio), audio_length_clean):
        audio_slice = audio[i:i+audio_length_clean]
        position = tfio.experimental.audio.trim(audio_slice, axis=0, epsilon=0.065)
        start, stop=position[0], position[1]
        if stop-start<5:
            continue
        audio_slice = audio_slice[start:stop]
        audio_clean = tf.concat([audio_clean, audio_slice], 0)

    if len(audio_clean)<audio_length*voice_max_length:
        audio = tf.concat([np.zeros([audio_length*voice_max_length-len(audio_clean)]), audio], 0)
    else:
        audio = audio[-(audio_length*voice_max_length):]

    spectrogram = tf.signal.stft(audio, frame_length=1024, frame_step=frame_step)
    spectrogram = (tf.math.log(tf.abs(tf.math.real(spectrogram)))/tf.math.log(tf.constant(10, dtype=tf.float32))*20)-60
    spectrogram = tf.where(tf.math.is_nan(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    spectrogram = tf.where(tf.math.is_inf(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    voice_length, voice = 0, []
    nb_part = len(audio)//audio_length
    part_length = len(spectrogram)//nb_part
    partsCount = len(range(0, len(spectrogram)-part_length, int(part_length/2)))
    parts = np.zeros((partsCount, part_length, 513))
    for i, p in enumerate(range(0, len(spectrogram)-part_length, int(part_length/2))):
        part = spectrogram[p:p+part_length]
        parts[i] = part
    return parts


testParts = audioToTensor('clips/common_voice_fr_19598904.wav')
print(testParts.shape)

def loadDataFromFile(filepath):
    dataVoice, dataString = [], []
    string_max_lenght = 0
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
        string_max_lenght = max(len(wordList), string_max_lenght)
        dataString.append(wordList)
        dataVoice.append(row[1].replace(".mp3", '.wav'))
    return dataVoice, dataString, string_max_lenght

dataVoice, dataString, string_max_lenght = loadDataFromFile('train.tsv')

print("voice_max_length: ", voice_max_length)
print("string_max_lenght: ", string_max_lenght)
tokenizer = Tokenizer(num_words=2000, lower=True, oov_token="<rare>")
tokenizer.fit_on_texts(dataString)
with io.open('tokenizer.txt', 'w', encoding='utf-8') as f:
    for word, index in tokenizer.word_index.items():
        f.write(word + ":" + str(index) + "\n")
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
def prepareData(dataString, dataVoice):
    X_voice, X_string, Y_string = list(), list(), list()
    for i, seq in enumerate(dataString):
        voice =  dataVoice[i]
        seq = tokenizer.texts_to_sequences([seq])[0]
        for j in range(1, len(seq)):
            in_seq, out_seq = seq[:j], seq[:j+1]
            in_seq = pad_sequences([in_seq], maxlen=string_max_lenght-1)[0]
            out_seq = pad_sequences([out_seq], maxlen=string_max_lenght-1)[0]
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
        return int(len(self.x_voice) / self.batch_size)
    def __getitem__(self, idx):
        batch_x_string = self.x_string[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_string = self.y_string[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_voice = []
        for i in range(0, batch_size):
            voice = audioToTensor('clips/' + self.x_voice[idx * self.batch_size + i])
            #print(tf.shape(voice))
            batch_x_voice.append(voice)
        return [np.array(batch_x_voice), np.array(batch_x_string)], np.array(batch_y_string)

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

if os.path.exists(model_name):
    print("Load: " + model_name)
    model = load_model(model_name)
else:
    latent_dim=64
    encoder_inputs = Input(shape=(testParts.shape[0], None, None, 1))
    preprocessing = TimeDistributed(preprocessing.Resizing(6, 129))(encoder_inputs)
    normalization = TimeDistributed(BatchNormalization())(preprocessing)
    conv2d = TimeDistributed(Conv2D(34, 3, activation='relu'))(normalization)
    conv2d = TimeDistributed(Conv2D(64, 3, activation='relu'))(conv2d)
    maxpool = TimeDistributed(MaxPooling2D())(conv2d)
    dropout = TimeDistributed(Dropout(0.25))(maxpool)
    flatten = TimeDistributed(Flatten())(dropout)
    encoder_lstm = LSTM(units=latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(flatten)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(string_max_lenght-1))
    dec_emb_layer = Embedding(vocab_size, latent_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_sentence.png', show_shapes=True)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

batch_size = 32
epochs = 30
model.fit(MySequence(X_voice, X_string, Y_string, batch_size), epochs=epochs, steps_per_epoch=len(X_string)//batch_size)
#model.save_weights(model_name+'.h5')
model.save(model_name)

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary(line_length=200)
tf.keras.utils.plot_model(encoder_model, to_file='model_encoder.png', show_shapes=True)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]
dec_emb2= dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_input)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_state_input, [decoder_outputs2] + decoder_states2)
decoder_model.summary(line_length=200)
tf.keras.utils.plot_model(decoder_model, to_file='model_decoder.png', show_shapes=True)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    decoded_sentence = "start"
    stop_condition = False
    while not stop_condition:
        sequence = tokenizer.texts_to_sequences([decoded_sentence.split(" ")])[0]
        sequence = pad_sequences([sequence], maxlen=string_max_lenght-1)
        sequence = np.array(sequence)
        output_tokens, h, c = decoder_model.predict([sequence] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = word_for_id(sampled_token_index, tokenizer)
        if(sampled_word==None):
            sampled_word = "index:" + str(sampled_token_index)
            sampled_token_index = 0
        decoded_sentence += ' ' + sampled_word
        if (sampled_word == 'end' or len(decoded_sentence.split(" ")) > string_max_lenght):
            stop_condition = True
    return decoded_sentence

print("Test voice recognition")
for test_path, test_string in [('clips/common_voice_fr_19598904.wav', "L'endroit est recouvert de goyaviers et d'acacias non endémiques"), ('clips/common_voice_fr_19598936.wav', 'La Foire sur la place décrit les début du compositeur à Paris'), ('clips/common_voice_fr_20268897.wav', 'Quand il revient à Paris, deux ans plus tard, la pépinière a été détruite'), ('clips/common_voice_fr_19999318.wav', "Peu de chansons de cet album n'ont pas connu de reprises"), ('clips/common_voice_fr_19733071.wav', 'La série avait été diffusée par la télévision algérienne quelques mois auparavant')]:
    print("test_string: ", test_string)
    test_voice = audioToTensor(test_path)
    print(np.array([test_voice]).shape)
    decoded_sentence = decode_sequence(np.array([test_voice]))
    print("decoded_sentence: ", decoded_sentence)
