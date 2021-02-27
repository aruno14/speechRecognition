import csv
import io
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import MeCab

#wakati = MeCab.Tagger("-Owakati -r /etc/mecabrc")
wakati = MeCab.Tagger("-Owakati")
maxWords = 4000
model_name = "model_sentence_ja"

latent_dim = 64
sampling_rate = 48000
frame_length = 1024#beccause SR is 48000
frame_time = 1024/48000
step_time = frame_time/2#10ms#0.008
image_width = 32#32*8+21=256+21=277ms
voice_max_length = int(7/(step_time*image_width))

fft_size = int(frame_length//2+1)
print("voice_max_length:", voice_max_length)

def audioToTensor(filepath:str):
    audio_binary = tf.io.read_file(filepath)
    audio, audioSR = tf.audio.decode_wav(audio_binary)
    audioSR = tf.get_static_value(audioSR)
    audio = tf.squeeze(audio, axis=-1)
    frame_step = int(audioSR * step_time)#0.008*48000=384
    block_length = int(frame_step * image_width + frame_length)
    max_freq = audioSR/2#48000/2 = 24000
    min_freq = (audioSR/(frame_length/2))#48000/(512/2) = 187
    res_freq = (audioSR/frame_length)/2#48000/512 = 93
    max_index = int(fft_size/(max_freq/1500))#let's keep frequencie below 800 -> 256/(24000/1000) = 32
    #print("max_index:", max_index)

    if len(audio)<block_length*voice_max_length:
        audio = tf.concat([np.zeros([block_length*voice_max_length - len(audio)]), audio], 0)
    else:
        audio = audio[-(block_length*voice_max_length):]

    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spectrogram = spectrogram[:,0:max_index]

    spectrogram = (tf.math.log(tf.abs(tf.math.real(spectrogram)))/tf.math.log(tf.constant(10, dtype=tf.float32))*20)-60
    spectrogram = tf.where(tf.math.is_nan(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    spectrogram = tf.where(tf.math.is_inf(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    partsCount = (len(spectrogram)-(image_width//2))//(image_width//2)
    #print(len(spectrogram), image_width//2, partsCount)
    parts = np.zeros((partsCount, image_width//2, max_index))
    for i, p in enumerate(range(0, partsCount)):
        p = i * (image_width//2)
        part = spectrogram[p:p+image_width]
        part = tf.expand_dims(part, axis=-1)
        resized_part = tf.image.resize(part, (image_width//2, max_index//2))#We resize all to be more efficient
        resized_part = tf.squeeze(resized_part, axis=-1)
        parts[i] = resized_part
    return parts

testParts = audioToTensor('clips/common_voice_ja_19482477.wav')
print("testParts.shape:", testParts.shape)

def loadDataFromFile(filepath:str):
    dataVoice, dataString, dataOriginal = [], [], []
    string_max_lenght = 0
    with open(filepath) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      next(reader)#skip header
      for row in reader:
        sentence = row[2].replace("。", "")
        wordList = wakati.parse("start " + sentence + " end").split()
        #if(len(wordList)<5):
        #    continue
        string_max_lenght = max(len(wordList), string_max_lenght)
        dataString.append(wordList)
        dataVoice.append(row[1].replace(".mp3", '.wav'))
        dataOriginal.append(sentence)
    return dataVoice, dataString, string_max_lenght, dataOriginal

dataVoice, dataString, string_max_lenght, dataOriginal = loadDataFromFile('train.tsv')

print("voice_max_length: ", voice_max_length)
print("string_max_lenght: ", string_max_lenght)
tokenizer = Tokenizer(num_words=maxWords, lower=True, oov_token="<rare>")
tokenizer.fit_on_texts(dataString)
with io.open('tokenizer.txt', 'w', encoding='utf-8') as f:
    for word, index in tokenizer.word_index.items():
        f.write(word + ":" + str(index) + "\n")

vocab_size = min(len(tokenizer.word_index) + 1, maxWords)
print('Vocabulary Size: %d' % vocab_size)

def serialize_example(voice, in_seq, out_seq, text):
  feature = {
      'voice': tf.train.Feature(float_list=tf.train.FloatList(value=voice.flatten())),
      'in_seq': tf.train.Feature(float_list=tf.train.FloatList(value=in_seq)),
      'out_seq': tf.train.Feature(float_list=tf.train.FloatList(value=out_seq.flatten())),
      'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf-8')]))
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def prepareData(dataString, dataVoice, dataOriginal, saveFile_path):
    writer = tf.io.TFRecordWriter(saveFile_path)
    count = 0
    for i, seq in enumerate(dataString):
        voice =  dataVoice[i]
        original = dataOriginal[i]
        seq = tokenizer.texts_to_sequences([seq])[0]
        for j in range(1, len(seq)):
            in_seq, out_seq = seq[:j], seq[:j+1]
            in_seq = pad_sequences([in_seq], maxlen=string_max_lenght-1)[0]
            out_seq = pad_sequences([out_seq], maxlen=string_max_lenght-1)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            Xvoice = audioToTensor('clips/' + voice)
            print(count, Xvoice.shape, in_seq.shape, out_seq.shape)
            writer.write(serialize_example(Xvoice, in_seq, out_seq, original))
            count+=1
    writer.close()

print("#Create tfrecords")
prepareData(dataString, dataVoice, dataOriginal, "train.tfrecords")
