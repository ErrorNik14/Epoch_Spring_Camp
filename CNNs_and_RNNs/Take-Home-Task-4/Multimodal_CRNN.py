import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # I wanted to ignore the early warning messages in the console, thus this line

import numpy as np
import pandas as pd
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, GlobalAvgPool2D, ReLU, SpatialDropout2D, GRU, LSTM, Bidirectional, Dense, TextVectorization, Embedding, Dropout, Concatenate
from tensorflow.keras.metrics import CategoricalAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall
import matplotlib.pyplot as plt
import seaborn
import sklearn
import librosa
import whisper
from scipy.signal import medfilt
from sklearn.metrics import confusion_matrix

transc_model = whisper.load_model('base') # Loading Whisper "base" model for transcribing the audio

emotions = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprised"}


# Gathering spectograms (without augmentation) for CNN model
X_spect = []
Y_spect = []
c=0
main_dir = "./ravdess emotional speech/"
for folder_name in os.listdir(main_dir):
    print("Checking folder",folder_name)
    sub_dir = os.path.join(main_dir,folder_name)
    for fn in os.listdir(sub_dir):
        file_dir = os.path.join(sub_dir,fn)
        if os.path.isfile(file_dir):
            y, sr = librosa.load(file_dir,sr=48000, mono=True) # Loading the audio, 1. with a SR=48 kHz and 2. only as a mono track
            y_p = librosa.util.fix_length(y, size=5*sr)        # 3. Padding to ensure duration=6 seconds
            # 4. Clearing noise values as much as possible, using an index threshold
            threshold = int(sr * 0.005)
            s, ph = librosa.magphase(librosa.stft(y_p))
            noise_p = np.mean(s[:, :threshold],axis=1)
            mask = (s > noise_p[:,None]).astype(float)
            mask = medfilt(mask, kernel_size=(1,5))
            s_clean = s * mask
            ph_clean = ph * mask
            y_clean = librosa.istft(s_clean * ph_clean)
            # 5. Retrieving the log-Mel spectogram of cleaned signal (in dB)
            mel_spect = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_fft=2048, hop_length=1600, n_mels=150)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            # 6. Normalising the spectogram
            mel_spect = (mel_spect - np.mean(mel_spect))/np.std(mel_spect)
            c+=1
            X_spect.append(mel_spect)
            label = int(fn.split("-")[2])
            Y_spect.append(label-1)
            print(f"{c}.", Y_spect[-1], '--->', emotions[Y_spect[-1]])

X_train_spect, X_test_spect, y_train_spect, y_test_spect = sklearn.model_selection.train_test_split(X_spect, Y_spect, test_size=0.2, random_state=42)

X_train_spect = np.array(X_train_spect).astype('float32')
X_test_spect = np.array(X_test_spect).astype('float32')

X_train_spect = np.expand_dims(X_train_spect, axis=-1)
X_test_spect = np.expand_dims(X_test_spect, axis=-1)

y_train_spect = tensorflow.keras.utils.to_categorical(y_train_spect, num_classes=8) 
y_test_spect = tensorflow.keras.utils.to_categorical(y_test_spect, num_classes=8)

np.savez("./CNN_data/training_data.npz", X=X_train_spect, y=y_train_spect)
np.savez("./CNN_data/testing_data.npz", X=X_test_spect, y=y_test_spect)


""" # Load CNN data
cnn_data = np.load('./CNN_data/training_data.npz')
X_train_spect = cnn_data['X']
y_train_spect = cnn_data['y']

cnn_test = np.load('./CNN_data/testing_data.npz')
X_test_spect = cnn_test['X']
y_test_spect = cnn_test['y'] """


# Gather transcription data from audio signals for RNN model
c=0
X_text = []
Y_text = []
main_dir = "./ravdess emotional speech/"
for folder_name in os.listdir(main_dir):
    print("Checking folder",folder_name)
    sub_dir = os.path.join(main_dir,folder_name)
    for fn in os.listdir(sub_dir):
        file_dir = os.path.join(sub_dir,fn)
        if os.path.isfile(file_dir):
            c+=1
            tran = transc_model.transcribe(file_dir) # Transcribing the audio
            tran = tran["text"].lower() # Converting transcribed text to lowercase
            if tran[-1] in "!@#$%^&*().,;:": # And removing any punctuation at the end
                tran = tran[:-1]
            X_text.append(tran)

            label = int(fn.split("-")[2])
            Y_text.append(label-1)

            print(f"{c}.", X_text[-1],"-", Y_text[-1], '--->', emotions[Y_text[-1]])

print("No. of transcripts=",c)

""" # Load RNN data
rnn_data = np.load("./RNN_data/proc_data.npz") 
X_text = rnn_data['X']
Y_text = rnn_data['y'] """

X_train_text, X_test_text, y_train_text, y_test_text = sklearn.model_selection.train_test_split(X_text, Y_text, test_size=0.2, random_state=42)

X_train_text = np.array(X_train_text).astype(object)
X_test_text = np.array(X_test_text).astype(object)

y_train_text = tensorflow.keras.utils.to_categorical(y_train_text, num_classes=8)
y_test_text = tensorflow.keras.utils.to_categorical(y_test_text, num_classes=8)

vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=200) # Text vectorization
vectorizer.adapt(X_train_text)

X_train_text_enc = vectorizer(X_train_text)
X_test_text_enc = vectorizer(X_test_text)


# Model definition!


# CNN portion
spect_input = tensorflow.keras.Input(shape=(150, 150, 1))
x1 = Conv2D(32, (3,3), activation='relu', padding='same')(spect_input)
x1 = BatchNormalization()(x1)
x1 = SpatialDropout2D(0.1)(x1)
x1 = Conv2D(64, (3,3), activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = SpatialDropout2D(0.3)(x1)
x1 = MaxPool2D()(x1)
x1 = Conv2D(128, (3,3), activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = SpatialDropout2D(0.3)(x1)
x1 = MaxPool2D()(x1)
x1 = GlobalAvgPool2D()(x1)
x1 = dense1 = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x1)
x1 = drop1 = Dropout(0.3)(x1)
x1 = dense2 = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x1)
x1 = drop2 = Dropout(0.3)(x1)

# RNN portion
text_input = tensorflow.keras.Input(shape=(200,))
x2 = Embedding(10000, 128)(text_input)
x2 = GRU(32)(x2)

# Multimodal portion
combined = Concatenate()([x1, x2])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(8, activation='softmax')(x)

model = tensorflow.keras.Model(inputs=[spect_input, text_input], outputs=output)

model.compile(optimizer=Adam(learning_rate=5e-4), 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy', 'f1_score'])

history = model.fit([X_train_spect, X_train_text_enc], y_train_text, batch_size=32, epochs=600, validation_split=0.2,
                    callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)])

model.evaluate([X_test_spect, X_test_text_enc], y_test_text, batch_size=32)


# Here is the code for creating a confusion matrix / heatmap of the model's predictions, to see where it went wrong 
# and what it confused.

predictions = model.predict([X_test_spect, X_test_text_enc])
predicted_class = tensorflow.argmax(predictions, axis=1)
actual_class = tensorflow.argmax(y_test_text, axis=1)

cm = tensorflow.math.confusion_matrix(labels=actual_class, predictions=predicted_class, num_classes=8)

print("Confusion Matrix:\n", cm.numpy())

plt.figure(figsize=(8, 6))
seaborn.heatmap(cm, annot=True, xticklabels=np.array(list(emotions.values())), yticklabels=np.array(list(emotions.values())))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()
plt.savefig("confusion_matrix_crnn.png")
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.savefig("epoch_vs_loss_crnn.png")
