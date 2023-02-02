import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, GRU, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

dataFolder="wordsFr"
words = ['bonjour', 'salut', 'merci', 'aurevoir']

latent_dim = 512
batch_size = 32
epochs = 64

block_length = 0.500#->500ms
audio_max_length = int(2/block_length)#->2s
frame_length = 512
step_length = 0.008
split_count = 10
fft_size = int(frame_length//2+1)
print("audio_max_length:", audio_max_length)

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
testParts = audioToTensor('wordsTestFr/bonjour-01.wav')
print(testParts.shape)
X_audio, Y_word = [], []

for i, word in enumerate(words):
    wordToId[word]=i
    idToWord[i]=word
    for file in glob.glob(dataFolder+"/"+word+'/*.wav'):
        audio = audioToTensor(file)
        X_audio.append(audio)
        Y_word.append(np.array(to_categorical([i], num_classes=len(words))[0]))

X_audio, Y_word = np.asarray(X_audio), np.asarray(Y_word)

X_audio, X_audio_test, Y_word, Y_word_test = train_test_split(X_audio, Y_word)
print("X_audio.shape: ", X_audio.shape)
print("Y_word.shape: ", Y_word.shape)
print("X_audio_test.shape: ", X_audio_test.shape)
print("Y_word_test.shape: ", Y_word_test.shape)

encoder_inputs = Input(shape=(testParts.shape[0], testParts.shape[1]))
#preprocessing = preprocessing.Resizing(6, 129)(encoder_inputs)
normalization = BatchNormalization()(encoder_inputs)
split = tf.keras.layers.Reshape((normalization.shape[1]//split_count, -1, normalization.shape[2], 1))(normalization)
print(split.shape)
conv2d = TimeDistributed(Conv2D(34, 3, activation='relu'))(split)
conv2d = TimeDistributed(Conv2D(64, 3, activation='relu'))(conv2d)
maxpool = TimeDistributed(MaxPooling2D())(conv2d)
dropout = TimeDistributed(Dropout(0.25))(maxpool)
flatten = TimeDistributed(Flatten())(dropout)
encoder_lstm = GRU(units=latent_dim)(flatten)
decoder_dense = Dense(len(words), activation='softmax')(encoder_lstm)
model = Model(encoder_inputs, decoder_dense)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])
model.summary()

tf.keras.utils.plot_model(model, to_file='model_wordFr.png', show_shapes=True)

history=model.fit(X_audio, Y_word, validation_data=(X_audio_test, Y_word_test), shuffle=True, batch_size=batch_size, epochs=epochs)
model.save("model_word_fr")

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['acc'])
plt.legend(['loss', 'acc'])
plt.savefig("learning-wordFr.png")
plt.show()
plt.close()

score = model.evaluate(X_audio_test, Y_word_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Test voice recognition")

for test_path, test_string in [('wordsTestFr/bonjour-01.wav', 'bonjour'), ('wordsTestFr/bonjour-011.wav', 'bonjour'), ('wordsTestFr/salut-01.wav', 'salut'), ('wordsTestFr/merci-01.wav', 'merci'), ('wordsTestFr/aurevoir-01.wav', 'aurevoir'), ('wordsTestFr/merci-011.wav', 'merci')]:
    print("test_string: ", test_string)
    test_audio = audioToTensor(test_path)
    result = model.predict(np.array([test_audio]))
    maxValue = np.argmax(result)
    print("decoded_sentence: ", result, maxValue, idToWord[maxValue])
