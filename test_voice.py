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

block_length = 0.050#->500ms
voice_max_length = int(1/block_length)#->2s
print("voice_max_length:", voice_max_length)
def arrayToTensor(audioArray, audioSR):
    audio = tf.convert_to_tensor(audioArray, dtype=tf.float32)/ 32768.0
    audioSR = tf.get_static_value(audioSR)
    audio_length = int(audioSR * block_length)#->16000*0.5=8000
    frame_step = int(audioSR * 0.008)#16000*0.008=128
    voice_length, audio_clean = 0, tf.constant([], tf.float32)
    audio_length_clean = audioSR//20
    if len(audio)<audio_length*voice_max_length:
        audio = tf.concat([np.zeros([audio_length*voice_max_length-len(audio)]), audio], 0)
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

model = tf.keras.models.load_model('model_words')
CHUNK = 1024*2
FORMAT = pyaudio.paInt32
CHANNELS = 1
RATE = 16000#44100
RECORD_SECONDS = 20
decoded_data = []
silenceCount = 0
audio_length = int(RATE * block_length)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("* recording")

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    count = len(data)/2
    format = "%dh"%(count)
    shorts = struct.unpack(format, data)
    even_i = []
    for j in range(0, len(shorts)):
        if j % 2:
            even_i.append(shorts[j])
    meanNoise = np.mean(np.abs(even_i))
    if(meanNoise < 300):#update this value if needed
        #print("silence:", meanNoise)
        silenceCount+=1
        if silenceCount > 5:
            decoded_data = []
        if silenceCount > 2:
            continue
    else:
        #print("Sound:", meanNoise)
        silenceCount=0
    decoded_data = decoded_data + even_i
    if(len(decoded_data) > audio_length*voice_max_length/2):
        test_audio = arrayToTensor(decoded_data, RATE)
        result = model.predict(np.array([test_audio]))[0]
        max = np.argmax(result)
        if(result[max]>0.5):
            print("finded word:", i, "->", result, max, result[max], idToWord[max])
        else:
            print("best candidate:", i, "->", result, max, result[max], idToWord[max])

print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()
