import numpy as np
import glob
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

dataFolder = "mini_speech_commands/"
words = [os.path.basename(x) for x in glob.glob(dataFolder + "*")]
words.remove("_background_noise_")
batch_size = 4
epochs = 16
block_length = 0.500#->500ms
audio_max_length = int(2/block_length)#->2s
frame_length = 512
fft_size = int(frame_length//2+1)
step_length = 0.008
split_count = 7
latent_dim=512

def audioToTensor(filepath:str):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    frame_step = int(audioSR * step_length)#16000*0.008=128
    if len(audio)<audioSR*audio_max_length:
        audio = tf.concat([np.zeros([int(audioSR*audio_max_length)-len(audio)]), audio], 0)
    else:
        audio = audio[-int(audioSR*audio_max_length):]
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_real = tf.math.real(spectrogram)
    spect_real = tf.abs(spect_real)
    spect_real = (tf.math.log(spect_real)/tf.math.log(tf.constant(10, dtype=tf.float32))*20)-60
    spect_real = tf.where(tf.math.is_nan(spect_real), tf.zeros_like(spect_real), spect_real)
    spect_real = tf.where(tf.math.is_inf(spect_real), tf.zeros_like(spect_real), spect_real)
    #spect_real = tf.image.resize(spect_real, (spect_real.shape[0]//2, spect_real.shape[1]//2))#We resize all to be more efficient
    return spect_real

wordToId, idToWord = {}, {}
testParts = audioToTensor(os.path.join(dataFolder, 'go/0a9f9af7_nohash_0.wav'))
print("Test", testParts.shape)

X_audio, Y_word = [], []
for i, word in enumerate(words):
    for file in glob.glob(os.path.join(dataFolder, word) + '/*.wav'):
        X_audio.append(file)
        Y_word.append(np.array(to_categorical([i], num_classes=len(words))[0]))

X_audio, Y_word = np.asarray(X_audio), np.asarray(Y_word)

class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_audio, y_word, batch_size):
        self.x_audio, self.y_word = x_audio, y_word
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x_audio) // self.batch_size

    def __getitem__(self, idx):
        batch_y = self.y_word[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((self.batch_size, testParts.shape[0], testParts.shape[1]))
        for i in range(0, batch_size): 
            batch_x[i] = audioToTensor(self.x_audio[idx * self.batch_size + i])
        return batch_x, batch_y

X_audio, X_audio_test, Y_word, Y_word_test = train_test_split(X_audio, Y_word)
print("X_audio.shape: ", X_audio.shape)
print("Y_word.shape: ", Y_word.shape)
print("X_audio_test.shape: ", X_audio_test.shape)
print("Y_word_test.shape: ", Y_word_test.shape)

encoder_inputs = Input(shape=(testParts.shape[0], testParts.shape[1]))
normalization = BatchNormalization()(encoder_inputs)
split = tf.keras.layers.Reshape((normalization.shape[1]//split_count, -1, normalization.shape[2], 1))(normalization)
conv2d = TimeDistributed(Conv2D(34, 3, activation='relu'))(split)
conv2d = TimeDistributed(Conv2D(64, 3, activation='relu'))(conv2d)
maxpool = TimeDistributed(MaxPooling2D())(conv2d)
dropout = TimeDistributed(Dropout(0.25))(maxpool)
flatten = TimeDistributed(Flatten())(dropout)
encoder_lstm = LSTM(units=latent_dim)(flatten)
decoder_dense = Dense(len(words), activation='softmax')(encoder_lstm)
model = Model(encoder_inputs, decoder_dense)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
tf.keras.utils.plot_model(model, to_file='model_words.png', show_shapes=True)

history=model.fit(MySequence(X_audio, Y_word, batch_size), shuffle=True, batch_size=batch_size, epochs=epochs, validation_data=MySequence(X_audio_test, Y_word_test, batch_size))
model.save("model_words")
metrics = history.history

plt.plot(history.epoch, metrics['loss'], metrics['acc'])
plt.legend(['loss', 'acc'])
plt.savefig("learning-words.png")
plt.show()
plt.close()

score = model.evaluate(X_audio_test, Y_word_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Test voice recognition")

for test_path, test_string in [('mini_speech_commands/go/0a9f9af7_nohash_0.wav', 'go'), ('mini_speech_commands/right/0c2ca723_nohash_0.wav', 'right')]:
    print("test_string: ", test_string)
    test_audio = audioToTensor(test_path)
    result = model.predict(np.array([test_audio]))
    maxIndex = np.argmax(result)
    print("decoded_sentence: ", result, maxIndex, idToWord[maxIndex])
