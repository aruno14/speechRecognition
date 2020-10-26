import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Embedding, Input, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

maxData = 10
dataString = []
string_max_lenght = 0
with open('validated.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    #header = (client_id path sentence up_votes down_votes age gender accent locale segment)
    next(reader)#skip header
    for row in reader:
        if len(dataString) > maxData:
            break
        sentence = ("start " + row[2] + " end").split(" ")
        print("sentence: ", sentence)
        dataString.append(sentence)
        string_max_lenght = max(len(sentence), string_max_lenght)
print("string_max_lenght: ", string_max_lenght)

tokenizer = Tokenizer(num_words=2000, lower=True, oov_token="<rare>")
tokenizer.fit_on_texts(dataString)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

X_voice, X_string, Y_string = list(), list(), list()
for i, seq in enumerate(dataString):
    seq_no_tag = seq[1:-1]
    seq = tokenizer.texts_to_sequences([seq])[0]
    seq_no_tag = tokenizer.texts_to_sequences([seq_no_tag])[0]
    seq_full_no_tag = pad_sequences([seq_no_tag], maxlen=string_max_lenght-2)[0]

    for j in range(1, len(seq)):
        in_seq, out_seq = seq[:j], seq[:j+1]
        in_seq = pad_sequences([in_seq], maxlen=string_max_lenght-1)[0]
        out_seq = pad_sequences([out_seq], maxlen=string_max_lenght-1)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

        X_voice.append(seq_full_no_tag)
        X_string.append(in_seq)
        Y_string.append(out_seq)

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

X_voice, X_string, Y_string = np.array(X_voice), np.array(X_string), np.array(Y_string)

latent_dim=32

print("x_voice.shape: ", X_voice.shape)
print("x_string.shape: ", X_string.shape)
print("y_string.shape: ", Y_string.shape)
num_encoder_tokens = X_voice.shape[1]
num_decoder_tokens = Y_string.shape[1]
print("num_encoder_tokens: ", num_encoder_tokens)
print("num_decoder_tokens: ", num_decoder_tokens)

# Set up the encoder
encoder_inputs = Input(shape=(num_encoder_tokens))
enc_emb =  Embedding(vocab_size, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(units=latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(num_decoder_tokens))
dec_emb_layer = Embedding(vocab_size, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary(line_length=200)
tf.keras.utils.plot_model(model, to_file='model_trad.png', show_shapes=True)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
batch_size = 32
epochs = 500
model.fit([X_voice, X_string], Y_string, epochs=epochs)

# Encode the input sequence to get the "Context vectors"
encoder_model = Model(encoder_inputs, encoder_states)
#encoder_model.save_weights('model_encoder.h5')
encoder_model.summary(line_length=200)
tf.keras.utils.plot_model(encoder_model, to_file='model_trad_encoder.png', show_shapes=True)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]
# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs)
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_state_input)
decoder_states2 = [state_h2, state_c2]
# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)
# Final decoder model
decoder_model = Model([decoder_inputs] + decoder_state_input, [decoder_outputs2] + decoder_states2)
#decoder_model.save_weights('model_decoder.h5')
decoder_model.summary(line_length=200)
tf.keras.utils.plot_model(decoder_model, to_file='model_trad_decoder.png', show_shapes=True)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    decoded_sentence = "start"
    stop_condition = False
    while not stop_condition:
        print("decoded_sentence: ", decoded_sentence)
        sequence = tokenizer.texts_to_sequences([decoded_sentence.split(" ")])[0]
        sequence = pad_sequences([sequence], maxlen=string_max_lenght-1)
        sequence = np.array(sequence)
        output_tokens, h, c = decoder_model.predict([sequence] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print("sampled_token_index: ", sampled_token_index)

        sampled_word = word_for_id(sampled_token_index, tokenizer)
        decoded_sentence += ' ' + sampled_word
        if (sampled_word == 'end' or len(decoded_sentence.split(" ")) > string_max_lenght):
            stop_condition = True
        # Update states
        #states_value = [h, c]
    return decoded_sentence

print("Test translation")
for test_string in ['She is eating']:
    print("test_string: ", test_string)
    wordList = ("start "+ test_string + " end").split(" ")
    print("wordList: ", wordList)
    in_seq = tokenizer.texts_to_sequences([wordList])[0]
    in_seq = pad_sequences([in_seq], maxlen=string_max_lenght-2)

    sequence = np.array(in_seq)
    decoded_sentence = decode_sequence(in_seq)
    print("decoded_sentence: ", decoded_sentence)
