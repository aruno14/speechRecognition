# SpeechRecognition
Speech recognition system implemented using *TensorFlow*.

## Files explanation
### test_load.py
Load labels file of **Mozilla Common Voice**:  https://commonvoice.mozilla.org/en/datasets

### test_lstm.py
Simple LSTM model to predict next words of a sequence of words. It uses Mozilla Common Voice dataset labels file.

### test_trad.py
Seq2Seq model to translate sentence in same language as the input. It uses Mozilla Common Voice dataset labels file.

### test_words.py
Simple LSTM model to convert audio into word. It uses **Speech command Dataset**: https://aiyprojects.withgoogle.com/open_speech_recording


### server_express.js
Simple WebServer in NodeJS.

### convert_mp3towav.sh
Convert all `mp3` files of current folder in `wav`.

### sentence_fr.py
Seq2Seq model to convert audio sentence in text. It uses Mozilla Common Voice French dataset.
