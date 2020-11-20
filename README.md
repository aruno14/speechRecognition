# SpeechRecognition
Speech recognition system implemented using *TensorFlow*.
More explanation in Medium article: https://aruno14.medium.com/a-journey-to-speech-recognition-using-tensorflow-1fc1169fef99

## Used libraries
```
pip install tensorflow tensorflow_io
```

## Files explanation
### test_load.py
Load labels file of **Mozilla Common Voice**:  https://commonvoice.mozilla.org/en/datasets

### test_lstm.py
Simple LSTM model to predict next words of a sequence of words. It uses Mozilla Common Voice dataset labels file.

### test_trad.py
Seq2Seq model to translate sentence in same language as the input. It uses Mozilla Common Voice dataset labels file.

### test_words.py
Simple LSTM model to convert audio into word. It uses **Speech command Dataset**: https://aiyprojects.withgoogle.com/open_speech_recording

#### test_wordsFr.py
Simple LSTM model to convert audio into French word. It uses **a self-made Dataset** included in this repository.

#### test_words_compare.py
Files to compare accuracy of different audio representation.
More explanation in Medium article: https://aruno14.medium.com/comparaison-of-audio-representation-in-tensorflow-b6c33a83d77f

#### test_words_human.py
Same as `test_words.py`, but high frequencies are cut.

### test_voice.py
Read audio input data from mic and predict word in real time.

### server_express.js
Simple WebServer in NodeJS.

### convert_mp3towav.sh
Convert all `mp3` files of current folder in `wav`.

### sentence_fr.py
Seq2Seq model to convert audio sentence in text. It uses Mozilla Common Voice French dataset.

## HTML pages
In order to avoid CORS probleme, we use a local WebServer. You can use `python3 -m http.server 3000` or `node server_express.js` and access: http://localhost:3000.

Model can recognize: `['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']`


In addition to English, there is also a French test page: http://localhost:3000/indexFr.html.

Model can recognize: `['bonjour', 'salut', 'merci', 'aurevoir']`

Below command has been used to convert model to TensorFlow.js:
```
~/.local/bin/tensorflowjs_converter model_reco/ quantized_model/ --input_format tf_saved_model --output_format tfjs_graph_model --quantize_float16
```
