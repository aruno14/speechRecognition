import pyaudio
import tensorflow as tf
import numpy as np
import struct

words =['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
wordToId = {}
idToWord = {}

for i, word in enumerate(words):
    wordToId[word] = i
    idToWord[i] = word

image_width =  128//4#128*0.008 = 1.024s
frame_length = 512
fft_size = int(frame_length//2+1)
step_length = 0.008
audio_max_length = 1.5#2s

def arrayToTensor(audioArray, audioSR:int):
    audio = tf.convert_to_tensor(audioArray, dtype=tf.float32)
    frame_step = int(audioSR * step_length)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_real = tf.math.real(spectrogram)
    spect_real = tf.abs(spect_real)
    spect_real = (tf.math.log(spect_real)/tf.math.log(tf.constant(10, dtype=tf.float32))*20)-60
    spect_real = tf.where(tf.math.is_nan(spect_real), tf.zeros_like(spect_real), spect_real)
    spect_real = tf.where(tf.math.is_inf(spect_real), tf.zeros_like(spect_real), spect_real)
    partsCount = len(spect_real)//(image_width//2)
    parts = np.zeros((partsCount, image_width, fft_size))
    for i, p in enumerate(range(0, len(spect_real)-image_width, image_width//2)):
        parts[i] = spect_real[p:p+image_width]
    return parts

model = tf.keras.models.load_model('model_words')
CHUNK = 16000
FORMAT, CHANNELS = pyaudio.paInt16, 1
RATE = 16000#44100
RECORD_SECONDS = 20
decoded_data = []
silenceCount = 0

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("* recording")

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    count = len(data)/2
    format = "%dh"%(count)
    shorts = struct.unpack(format, data)
    newData = np.asarray(shorts)/32767
    meanNoise = np.mean(np.abs(newData))
    #print(meanNoise, np.max(newData))
    if(meanNoise < 0.001):#update this value if needed
        #print("silence:", meanNoise)
        silenceCount+=1
        if silenceCount > 5:
            decoded_data = []
        if silenceCount > 2:
            continue
    else:
        print("Sound:", meanNoise)
        silenceCount=0
    decoded_data = np.concatenate([decoded_data, newData], axis=-1)
    if(len(decoded_data) >= int(RATE*audio_max_length)):
        input_data = decoded_data[-int(RATE*audio_max_length):]
        output_audio = arrayToTensor(input_data, RATE)
        output_audio = tf.expand_dims(output_audio, axis=-1)
        result = model.predict(np.array([output_audio]))[0]
        max = np.argmax(result)
        if(result[max]>0.5):
            print("finded word:", i, "->", result, max, result[max], idToWord[max])
        else:
            print("best candidate:", i, "->", result, max, result[max], idToWord[max])

print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()
