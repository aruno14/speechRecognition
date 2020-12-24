import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import matplotlib.pyplot as plt

words=['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
batch_size = 64
epochs = 50
frame_length = 512
fft_size = int(frame_length//2+1)
step_length = 0.008
image_width = 128//4#128*0.008 = 1.024s
audio_max_length = 1.5#1.5

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
    partsCount = len(spect_real)//(image_width//2)
    parts = np.zeros((partsCount, image_width//2, fft_size//2))
    for i, p in enumerate(range(0, len(spect_real)-image_width, image_width//2)):
        part = spect_real[p:p+image_width]
        part = tf.expand_dims(part, axis=-1)
        resized_part = tf.image.resize(part, (image_width//2, fft_size//2))#We resize all to be more efficient
        resized_part = tf.squeeze(resized_part, axis=-1)
        parts[i] = resized_part
    return parts

max_data = 900
wordToId, idToWord = {}, {}
testParts = audioToTensor('mini_speech_commands/go/0a9f9af7_nohash_0.wav')
print(testParts.shape)
X_audio, Y_word = np.zeros((max_data*8, testParts.shape[0], testParts.shape[1], testParts.shape[2])), np.zeros((max_data*8, 8))

files = {}
for i, word in enumerate(words):
    wordToId[word], idToWord[i] = i, word
    files[word] = glob.glob('mini_speech_commands/'+word+'/*.wav')

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
preprocessing = TimeDistributed(preprocessing.Resizing(testParts.shape[1]//2, testParts.shape[2]//2))(encoder_inputs)
normalization = TimeDistributed(BatchNormalization())(preprocessing)
conv2d = TimeDistributed(Conv2D(34, 3, activation='relu'))(normalization)
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

history=model.fit(X_audio, Y_word, shuffle=False, batch_size=batch_size, epochs=epochs, steps_per_epoch=len(X_audio)//batch_size, validation_data=(X_audio_test, Y_word_test))
model.save_weights('model_words.h5')
model.save("model_words")
metrics = history.history

plt.plot(history.epoch, metrics['loss'], metrics['acc'])
plt.legend(['loss', 'acc'])
plt.savefig("learning-words.png")
plt.show()
plt.close()

score = model.evaluate(X_audio_test, Y_word_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("Test voice recognition")

for test_path, test_string in [('mini_speech_commands/go/0a9f9af7_nohash_0.wav', 'go'), ('mini_speech_commands/right/0c2ca723_nohash_0.wav', 'right')]:
    print("test_string: ", test_string)
    test_audio = audioToTensor(test_path)
    result = model.predict(np.array([test_audio]))
    max = np.argmax(result)
    print("decoded_sentence: ", result, max, idToWord[max])
