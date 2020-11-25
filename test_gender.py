import csv
import io
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

maxData = 500
model_name = "model_gender"
block_length = 0.5
frame_length = 1024
image_width = 128

def audioToTensor(filepath):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    frame_step = int(audioSR * 0.008)
    if len(audio) < frame_step*(image_width+2) + frame_length:
        audio = tf.concat([np.zeros([(frame_step*(image_width+2)+frame_length)-len(audio)]), audio], 0)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_real = tf.math.real(spectrogram)
    spect_real = tf.abs(spect_real)
    partsCount = len(range(0, len(spectrogram)-image_width, image_width))
    parts = np.zeros((partsCount, image_width, int(frame_length/2+1)))
    for i, p in enumerate(range(0, len(spectrogram)-image_width, image_width)):
        part = spect_real[p:p+image_width]
        parts[i] = part
    return parts, audioSR

def loadDataFromFile(filepath):
    dataVoice_male, dataGender_male, dataVoice_female, dataGender_female = [], [], [], []
    with open(filepath) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      next(reader)#skip header
      for row in reader:
        if row[6] == "":
            continue
        if min(len(dataGender_female), len(dataGender_male))>maxData:
            break
        filename = row[1].replace('.mp3', '.wav')
        if row[6] == "male":
            gender = [1, 0]
            dataGender_male.append(gender)
            dataVoice_male.append(filename)
        else:
            gender = [0, 1]
            dataGender_female.append(gender)
            dataVoice_female.append(filename)
    return dataVoice_male, dataGender_male, dataVoice_female, dataGender_female

dataVoice_male, dataGender_male, dataVoice_female, dataGender_female = loadDataFromFile('validated.tsv')
print("len(dataGender_male):", len(dataGender_male))
print("len(dataGender_female):", len(dataGender_female))

dataVoice, dataGender = [], []
parts_count = 0
min_parts = 100
for i in range(0, min(len(dataGender_female), len(dataGender_male))):
    dataVoice.append(dataVoice_male[i])
    dataVoice.append(dataVoice_female[i])
    dataGender.append(dataGender_male[i])
    dataGender.append(dataGender_female[i])
    parts1 = len(audioToTensor('clips/' + dataVoice_male[i]))
    parts2 = len(audioToTensor('clips/' + dataVoice_female[i]))
    parts_count += parts1
    parts_count += parts2
    min_parts = min(min_parts, parts1, parts2)

print("parts_count:", parts_count)
print("min_parts:", min_parts)

class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_voice, y_gender, batch_size, parts_count, min_parts):
        self.x_voice, self.y_gender = x_voice, y_gender
        self.batch_size = batch_size
        self.parts_count = parts_count
        self.min_parts = min_parts
    def __len__(self):
        return (len(self.x_voice)*self.min_parts) // self.batch_size
    def __getitem__(self, idx):
        batch_x_voice = np.zeros((batch_size, image_width, int(frame_length/2+1)))
        batch_y_gender = np.zeros((batch_size, 2))
        for i in range(0, batch_size//self.min_parts):
            gender = self.y_gender[idx * self.batch_size//self.min_parts + i]
            voice, _ = audioToTensor('clips/' + self.x_voice[idx * self.batch_size//self.min_parts + i])
            for j in range(0, self.min_parts):
                batch_x_voice[i*self.min_parts+j] = random.choice(voice)
                batch_y_gender[i*self.min_parts+j] = gender
        return batch_x_voice, batch_y_gender

if os.path.exists(model_name):
    print("Load: " + model_name)
    model = load_model(model_name)
else:
    main_input = Input(shape=(image_width, int(frame_length/2+1)), name='main_input')
    x = main_input
    x = Reshape((image_width, int(frame_length/2+1), 1))(x)
    x = preprocessing.Resizing(image_width//2, int(frame_length/2+1)//2)(x)
    x = Conv2D(34, 3, activation='relu')(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=x)
    tf.keras.utils.plot_model(model, to_file='model_gender.png', show_shapes=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

batch_size = 32
epochs = 1
history = model.fit(MySequence(dataVoice[0:int(len(dataVoice)*0.8)], dataGender[0:int(len(dataVoice)*0.8)], batch_size, parts_count, min_parts), epochs=epochs, validation_data=MySequence(dataVoice[int(len(dataVoice)*0.8):], dataGender[int(len(dataVoice)*0.8):], batch_size, parts_count, min_parts))
model.save(model_name)

metrics = history.history

plt.plot(history.epoch, metrics['loss'], metrics['acc'])
plt.legend(['loss', 'acc'])
plt.savefig("learning-gender.png")
plt.show()
plt.close()

print("Test voice gender recognition")
for test_path in ['wordsTestFr/bonjour-01.wav', 'wordsTestFr/bonjour-011.wav', 'wordsTestFr/salut-01.wav', 'wordsFr/bonjour/bonjour-11.wav', 'yuki2.wav']:
    print("test_path: ", test_path)
    test_voice, _ = audioToTensor(test_path)
    print(test_voice.shape)
    predictions = model.predict(np.asarray(test_voice))
    print("predictions: ", predictions)
