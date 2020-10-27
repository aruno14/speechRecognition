import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import matplotlib.pyplot as plt

words = ['bonjour', 'salut', 'merci', 'aurevoir']
block_length = 0.050#->500ms
voice_max_length = int(1/block_length)#->2s
print("voice_max_length:", voice_max_length)

def audioToTensor(filepath):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)#WebAudioAPI seems to return -1~1
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    audio_lenght = int(audioSR * block_length)#->16000*0.5=8000
    frame_step = int(audioSR * 0.008)#16000*0.008=128 when rate is 1600 -> 8ms
    if len(audio)<audio_lenght*voice_max_length:
        audio = tf.concat([np.zeros([audio_lenght*voice_max_length-len(audio)]), audio], 0)
    else:
        audio = audio[-(audio_lenght*voice_max_length):]
    spectrogram = tf.signal.stft(audio, frame_length=1024, frame_step=frame_step)
    spectrogram = (tf.math.log(tf.abs(tf.math.real(spectrogram)))/tf.math.log(tf.constant(10, dtype=tf.float32))*20)-60
    spectrogram = tf.where(tf.math.is_nan(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    spectrogram = tf.where(tf.math.is_inf(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    voice_length, voice = 0, []
    nb_part = len(audio)//audio_lenght
    part_length = len(spectrogram)//nb_part
    partsCount = len(range(0, len(spectrogram)-part_length, int(part_length/2)))
    parts = np.zeros((partsCount, part_length, 513))
    for i, p in enumerate(range(0, len(spectrogram)-part_length, int(part_length/2))):
        part = spectrogram[p:p+part_length]
        parts[i] = part
    return parts

max_data = 200
wordToId, idToWord = {}, {}

testParts = audioToTensor('wordsTestFr/bonjour-01.wav')
print(testParts.shape)

files = {}
for i, word in enumerate(words):
    wordToId[word], idToWord[i] = i, word
    files[word] = glob.glob('wordsFr/'+word+'/*.wav')

max_data = min(len(files["bonjour"]), max_data)
X_audio, Y_word = np.zeros((max_data*len(words), testParts.shape[0], testParts.shape[1], testParts.shape[2])), np.zeros((max_data*len(words), len(words)))
for nb in range(0, max_data):
    for i, word in enumerate(words):
        audio = audioToTensor(files[word][nb])
        X_audio[len(files)*nb + i] = audio
        Y_word[len(files)*nb + i] = np.array(to_categorical([i], num_classes=len(words))[0])

X_audio_test, Y_word_test = X_audio[int(len(X_audio)*0.8):], Y_word[int(len(Y_word)*0.8):]
X_audio, Y_word = X_audio[:int(len(X_audio)*0.8)], Y_word[:int(len(Y_word)*0.8)]
print("X_audio.shape: ", X_audio.shape)
print("Y_word.shape: ", Y_word.shape)
print("X_audio_test.shape: ", X_audio_test.shape)
print("Y_word_test.shape: ", Y_word_test.shape)

latent_dim=32
encoder_inputs = Input(shape=(testParts.shape[0], None, None, 1))
preprocessing = TimeDistributed(preprocessing.Resizing(6, 129))(encoder_inputs)
normalization = TimeDistributed(BatchNormalization())(preprocessing)
conv2d = TimeDistributed(Conv2D(34, 3, activation='relu'))(normalization)
conv2d = TimeDistributed(Conv2D(64, 3, activation='relu'))(conv2d)
maxpool = TimeDistributed(MaxPooling2D())(conv2d)
dropout = TimeDistributed(Dropout(0.25))(maxpool)
flatten = TimeDistributed(Flatten())(dropout)
encoder_lstm = LSTM(units=latent_dim)(flatten)
decoder_dense = Dense(len(words), activation='softmax')(encoder_lstm)

model = Model(encoder_inputs, decoder_dense)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])
model.summary(line_length=200)

tf.keras.utils.plot_model(model, to_file='model_wordFr.png', show_shapes=True)

batch_size = 32
epochs = 100
history=model.fit(X_audio, Y_word, shuffle=False, batch_size=batch_size, epochs=epochs, steps_per_epoch=len(X_audio)*1.0//batch_size, validation_data=(X_audio_test, Y_word_test))
model.save_weights('model_word_fr.h5')
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
    max = np.argmax(result)
    print("decoded_sentence: ", result, max, idToWord[max])
