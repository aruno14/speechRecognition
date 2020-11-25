import csv
import io
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

maxData = 300
model_name = "model_age"
block_length = 0.5
frame_length = 1024
image_width = 128
classesCount=7

def audioToTensor(filepath):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    frame_step = int(audioSR * 0.008)
    if len(audio) < frame_step*(image_width+3):
        audio = tf.concat([np.zeros([(frame_step*(image_width+3))-len(audio)]), audio], 0)
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
    dataVoice, dataGender = [], []
    weights = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    with open(filepath) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      next(reader)#skip header
      for row in reader:
        if row[5] == "":
              continue
        if len(dataGender)>maxData:
            break
        filename = row[1].replace('.mp3', '.wav')

        if row[5] == "teens":
            gender = [1, 0, 0, 0, 0, 0, 0]
            weights[0]+=1
        elif row[5] == "twenties":
            gender = [0, 1, 0, 0, 0, 0, 0]
            weights[1]+=1
        elif row[5] == "thirties":
            gender = [0, 0, 1, 0, 0, 0, 0]
            weights[2]+=1
        elif row[5] == "fourties":
            gender = [0, 0, 0, 1, 0, 0, 0]
            weights[3]+=1
        elif row[5] == "fifties":
            gender = [0, 0, 0, 0, 1, 0, 0]
            weights[4]+=1
        elif row[5] == "sixties":
            gender = [0, 0, 0, 0, 0, 1, 0]
            weights[5]+=1
        elif row[5] == "seventies":
            gender = [0, 0, 0, 0, 0, 0, 1]
            weights[6]+=1
        dataGender.append(gender)
        dataVoice.append(filename)
    return dataVoice, dataGender, weights

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
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

if os.path.exists(model_name):
    print("Load: " + model_name)
    model = load_model(model_name, custom_objects={'age_mae':age_mae})
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
    x = Dense(classesCount, activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=x)
    tf.keras.utils.plot_model(model, to_file='model_age.png', show_shapes=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[age_mae])

batch_size = 32
epochs = 15
history = model.fit(MySequence(dataVoice, dataAge, batch_size, parts_count, min_parts), epochs=epochs, steps_per_epoch=(len(dataVoice)*min_parts) // batch_size, class_weight=weights)
model.save(model_name)

metrics = history.history

plt.plot(history.epoch, metrics['loss'], metrics['age_mae'])
plt.legend(['loss', 'age_mae'])
plt.savefig("learning-age.png")
plt.show()
plt.close()

print("Test voice gender recognition")
for test_path in ['wordsTestFr/bonjour-01.wav', 'wordsTestFr/bonjour-011.wav', 'wordsTestFr/salut-01.wav']:
    print("test_path: ", test_path)
    test_voice, _ = audioToTensor(test_path)
    predictions = model.predict(np.asarray(test_voice))
    age = 0
    for i, p in enumerate(predictions[0]):
        age+= p * (i+1) * 10
    print("predictions: ", predictions)
    print("age:", age)
