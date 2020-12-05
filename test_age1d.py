import csv
import io
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv1D, MaxPooling1D, Dropout, Flatten, Reshape, AveragePooling1D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

maxData = 30000
model_name = "model_age1d"
frame_length = 1024
spect_length = int(frame_length/2+1)
step_time = 0.008
image_width = 100#100*0.008=800ms
classesCount = 7
batch_size = 64
epochs = 15

def audioToTensor(filepath):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    frame_step = int(audioSR * step_time)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_real = tf.math.real(spectrogram)
    spect_real = tf.abs(spect_real)
    partsCount = len(spect_real)//image_width
    parts = np.zeros((partsCount, image_width, spect_length))
    for i, p in enumerate(range(0, len(spectrogram)-image_width, image_width)):
        parts[i] = spect_real[p:p+image_width]
    return parts, audioSR

def loadDataFromFile(filepath):
    dataVoice, dataAge = [], []
    weights = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    with open(filepath) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      next(reader)#skip header
      for row in reader:
        if row[5] == "":
              continue
        if len(dataAge)>maxData:
            break
        filename = row[1].replace('.mp3', '.wav')

        if row[5] == "teens":
            age = [1, 0, 0, 0, 0, 0, 0]
            weights[0]+=1
        elif row[5] == "twenties":
            age = [0, 1, 0, 0, 0, 0, 0]
            weights[1]+=1
        elif row[5] == "thirties":
            age = [0, 0, 1, 0, 0, 0, 0]
            weights[2]+=1
        elif row[5] == "fourties":
            age = [0, 0, 0, 1, 0, 0, 0]
            weights[3]+=1
        elif row[5] == "fifties":
            age = [0, 0, 0, 0, 1, 0, 0]
            weights[4]+=1
        elif row[5] == "sixties":
            age = [0, 0, 0, 0, 0, 1, 0]
            weights[5]+=1
        elif row[5] == "seventies":
            age = [0, 0, 0, 0, 0, 0, 1]
            weights[6]+=1
        dataAge.append(age)
        dataVoice.append(filename)
    return dataVoice, dataAge, weights

dataVoice, dataAge, weights= loadDataFromFile('validated.tsv')
print("len(dataAge):", len(dataAge))
print("weights:", weights)

parts_count = 0
min_parts = 100
for i in range(0, len(dataAge)):
    parts1 = len(audioToTensor('clips/' + dataVoice[i]))
    parts_count += parts1
    min_parts = min(min_parts, parts1)

print("parts_count:", parts_count)
print("min_parts:", min_parts)

class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_voice, y_age, batch_size, parts_count, min_parts):
        self.x_voice, self.y_age = x_voice, y_age
        self.batch_size = batch_size
        self.parts_count = parts_count
        self.min_parts = min_parts
    def __len__(self):
        return (len(self.x_voice)*self.min_parts) // self.batch_size
    def __getitem__(self, idx):
        batch_x_voice = np.zeros((batch_size, image_width, int(frame_length/2+1)))
        batch_y_age = np.zeros((batch_size, classesCount))
        for i in range(0, batch_size//self.min_parts):
            #print(idx, self.batch_size, self.min_parts, i, idx * self.batch_size//self.min_parts + i)
            age = self.y_age[idx * self.batch_size//self.min_parts + i]
            voice, _ = audioToTensor('clips/' + self.x_voice[idx * self.batch_size//self.min_parts + i])
            for j in range(0, self.min_parts):
                batch_x_voice[i*self.min_parts+j] = random.choice(voice)
                batch_y_age[i*self.min_parts+j] = age
        return batch_x_voice, batch_y_age

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, classesCount, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, classesCount, dtype="float32"), axis=-1)
    mae = K.abs(true_age - pred_age)*10
    return mae

if os.path.exists(model_name):
    print("Load: " + model_name)
    model = load_model(model_name, custom_objects={'age_mae':age_mae})
else:
    main_input = Input(shape=(image_width, int(frame_length/2+1)), name='main_input')
    x = main_input
    x = BatchNormalization()(x)
    x = Conv1D(8, 3, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = AveragePooling1D(pool_size=3, strides=3)(x)
    x = Dropout(0.1)(x)
    x = Flatten(name="flatten")(x)
    x = Dense(classesCount, activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=x)
    tf.keras.utils.plot_model(model, to_file=model_name+'.png', show_shapes=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[age_mae])

history = model.fit(MySequence(dataVoice[:int(len(dataVoice)*0.8)], dataAge[:int(len(dataVoice)*0.8)], batch_size, parts_count, min_parts), epochs=epochs, class_weight=weights, validation_data=MySequence(dataVoice[int(len(dataVoice)*0.8):], dataAge[int(len(dataVoice)*0.8):], batch_size, parts_count, min_parts))

model.save(model_name)
