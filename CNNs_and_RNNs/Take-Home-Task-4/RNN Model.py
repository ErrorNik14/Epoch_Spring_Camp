import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # I wanted to ignore the early warning messages in the console, thus this line

import numpy as np
import pandas as pd
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, TextVectorization, Embedding, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy, Recall, Precision
import matplotlib.pyplot as plt
import seaborn
import sklearn
import librosa
import whisper
from sklearn.metrics import confusion_matrix

transc_model = whisper.load_model('base') # Loading Whisper "base" model for transcribing the audio

emotions = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprised"}


# Defining the model
class SemanticAnalyser(tensorflow.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = TextVectorization(max_tokens=10000, output_sequence_length=200)
        self.embed = Embedding(input_dim=10000, output_dim=128, mask_zero=True)
        
        self.gru = GRU(128, return_sequences=False)
        self.dense1 = Dense(64, activation='relu')
        self.drop1 = Dropout(0.3)
        self.dense2 = Dense(8, activation='softmax')
        


    def call(self, input):  
        x = self.encoder(input)
        x = self.embed(x)
        x = self.gru(x)
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        return x


    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Getting transcriptions from all the audio samples
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
            tran = transc_model.transcribe(file_dir) # Transcribing the audio
            tran = tran["text"].lower() # Converting transcribed text to lowercase
            if tran[-1] in "!@#$%^&*().,;:": # And removing any punctuation at the end
                tran = tran[:-1]
            X.append(tran)

            label = int(fn.split("-")[2])
            Y.append(label-1)

            print(f"{c}.", X[-1],"-", Y[-1], '--->', emotions[Y[-1]])

print("No. of transcripts=",c)
# np.savez("./RNN_data/proc_data.npz",X=np.array(X), y=np.array(Y))

# # Loading the pre-processed data
# proc_data = np.load("./RNN_data/proc_data.npz") 
# X = proc_data['X']
# Y = proc_data['y']

# Train-test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = np.array(X_train).astype(object)
X_test = np.array(X_test).astype(object)

 # One hot encoding the label (training and testing)
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=8)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=8)

model = SemanticAnalyser()
model.encoder.adapt(X_train) # Adapting the encoder vocabulary to our training set

callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

model.compile(optimizer=Adam(learning_rate=4e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy', 'f1_score'])

history = model.fit(X_train, y_train, batch_size=32, epochs=300, validation_split=0.2, shuffle=True, callbacks=[callback])

model.evaluate(X_test, y_test, batch_size=32)

model.save('rnn_model_alpha.keras')

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
# plt.show()
plt.savefig("confusion_matrix_rnn.png")
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.savefig("epoch_vs_loss_rnn.png")
