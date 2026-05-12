import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # I wanted to ignore the early warning messages in the console, thus this line

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, AvgPool2D, BatchNormalization, GlobalAvgPool2D, ReLU, Dropout, SpatialDropout2D
from tensorflow.keras.metrics import CategoricalAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall
import librosa
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.metrics import confusion_matrix
from scipy.signal import medfilt


emotions = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprised"}

# Some augmentation functions

def add_white_noise(y, n_factor=None):
    if n_factor is None:
        n_factor = np.random.uniform(0.005, 0.02)
    noise = np.random.normal(0, y.std(), y.size)
    return y + n_factor * noise

def pitch_scale(y, sr, n_semitones=None):
    if n_semitones is None:
        n_semitones = np.random.uniform(-3, 3)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones)

def random_gain(y, min_gain=0.7, max_gain=1.3):
    if max_gain < min_gain:
        max_gain, min_gain = min_gain, max_gain
    gain_factor = np.random.uniform(min_gain, max_gain)
    return y * gain_factor


# Defining our CNN model
'''
Model architecture
'''
class SemanticAnalyser(tensorflow.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = Conv2D(32, (3,3), activation='relu', padding='same')
        self.batchnorm1 = BatchNormalization()
        self.spatialdropout1 = SpatialDropout2D(0.2)

        self.conv2 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.batchnorm2 = BatchNormalization()
        self.spatialdropout2 = SpatialDropout2D(0.3)
        self.maxpool1 = MaxPool2D()


        self.conv3 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.batchnorm3 = BatchNormalization()
        self.spatialdropout3 = SpatialDropout2D(0.3)
        
        self.maxpool2 = MaxPool2D()


        self.globalavgpool = GlobalAvgPool2D()
        self.dense1 = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))
        self.drop1 = Dropout(0.3)
        self.dense2 = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))
        self.drop2 = Dropout(0.3)
        self.dense3 = Dense(8, activation='softmax')


    def call(self, input):  
        x = self.conv1(input)
        x = self.batchnorm1(x)
        x = self.spatialdropout1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.spatialdropout2(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.spatialdropout3(x)

        x = self.maxpool2(x)

        x = self.globalavgpool(x)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.dense3(x)
        return x


    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Audio pre-processing
'''My goal is pretty straightforward for the pre-processing.
    1. Resample it to a uniform sampling rate (48 kHz)
    2. Ensure audio is in mono-track
    3. Pad it out to a uniform length (5 seconds)
    4. Remove noise frequencies as much as possible
    5. Convert audio data into Mel-Spectograms
    6. Z-Score normalisation of Mel-Spectogram'''
# We start off by accessing the audio files, in the ravdess emotional speech dataset (extracted from zip)
c=0
X = []
Y = []
main_dir = "./ravdess emotional speech/"
for folder_name in os.listdir(main_dir):
    print("Checking folder",folder_name)
    sub_dir = os.path.join(main_dir,folder_name)
    for fn in os.listdir(sub_dir):
        file_dir = os.path.join(sub_dir,fn)
        if os.path.isfile(file_dir):
            c+=1
            X.append(file_dir) # Adding the file to an array for splitting and storing purposes

             # Retrieving the corresponding label (the 3rd number in the file name)
            label = int(fn.split("-")[2])
            Y.append(label-1)


train_files, test_files, y_train1, y_test1 = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

X_train = []
X_test = []

y_train = []
y_test = []

c=0
for i in range(len(train_files)):
    y, sr = librosa.load(train_files[i],sr=48000, mono=True) # Loading the audio, 1. with a SR=48 kHz and 2. only as a mono track
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

    # This is the part for Data Augmentation. I am going to add white noise, pitch scaling, random gain,
    # to try and make the model more robust.
    aug_y = [y_clean, add_white_noise(y_clean, np.random.random()*0.3), pitch_scale(y_clean, sr, np.random.randint(10,30)), 
                        random_gain(y_clean, np.random.random()*0.2, np.random.random()*0.8)]
    j=0
    for sig in aug_y: # Adding all augmented spectograms
        # 5. Retrieving the log-Mel spectogram of cleaned signal (in dB)
        mel_spect = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=2048, hop_length=1600, n_mels=150)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        # 6. Normalising the spectogram
        mel_spect = (mel_spect - np.mean(mel_spect))/np.std(mel_spect)

        X_train.append(mel_spect)
        y_train.append(y_train1[i])
        print(f"{c}.", y_train[-1], '--->', emotions[y_train[-1]], "map=",j)
        j+=1
    c+=1
print(f"No. of training audio files to process = {c}x4")

c=0
for i in range(len(test_files)):
    y, sr = librosa.load(test_files[i],sr=48000, mono=True) # Loading the audio, 1. with a SR=48 kHz and 2. only as a mono track
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

    X_test.append(mel_spect)
    y_test.append(y_test1[i])
    print(f"{c}.", y_test[-1], '--->', emotions[y_test[-1]])
    c+=1

print(f"No. of testing audio files to process = {c}")


X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=8)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=8)

""" np.savez("./augment_data/training_data.npz", X=X_train, y=y_train)
np.savez("./augment_data/testing_data.npz", X=X_test, y=y_test) """
# I used the numpy savez and load functions above, as I wanted to store the pre-processed datasets for 2 reason
        # 1. Not have to deal with any potential errors of the pre-processing state while working on the CNN model
        # 2. Access them as many times as needed while debugging the CNN model, without waiting more, thus speeding up the process

""" training_data = np.load('./augment_data/training_data.npz')
X_train = training_data['X']
y_train = training_data['y']

testing_data = np.load('./augment_data/testing_data.npz')
X_test = testing_data['X']
y_test = testing_data['y'] """

callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)

model = SemanticAnalyser()
model.compile(optimizer=Adam(5e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy', 'f1_score'])

history = model.fit(X_train, y_train, batch_size=32, epochs=300, validation_split=0.2, shuffle=True, callbacks=[callback])

model.evaluate(X_test, y_test, batch_size=32)

# Here is the code for creating a confusion matrix / heatmap of the model's predictions, to see where it went wrong 
# and what it confused.

predictions = model.predict(X_test)
predicted_class = tensorflow.argmax(predictions, axis=1)
actual_class = tensorflow.argmax(y_test, axis=1)

cm = tensorflow.math.confusion_matrix(labels=actual_class, predictions=predicted_class, num_classes=8)

print("Confusion Matrix:\n", cm.numpy())

plt.figure(figsize=(8, 6))
seaborn.heatmap(cm, annot=True, xticklabels=np.array(list(emotions.values())), yticklabels=np.array(list(emotions.values())))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix_cnn.png")
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.savefig("epoch_vs_loss_cnn.png")
