## Epoch Spring Camp 2026

#### Take Home Assignment-4

#### Name: Nikhil S Ramcharan

GitHub username: ErrorNik14 (sleepynik07)

### CNN Model (Spectrogram Analysis)
I initially took some inspiration from the LeNet Architecture for this model, adding another Convolutional layer to make the feature map even smaller for the Dense layer analysis. The values for dropout and L2 regulariser were determined after a number of runs and experimentations.

I applied data augmentation to the training audio, feeling that the 1440 audio samples provided by the RAVDESS dataset would be too small to meaningfully train the model. For each sample, I created one copy with random white noise, another copy with random audio gain, and another copy with random pitch shift. I was able to increase the available training samples from 1152 samples to ~4608 samples, greatly improving the information I could extract from the dataset.

I used the given below architecture, to be precise.



My training/validation plot looked like this



It is apparent how after the 60th epoch or so, overfitting started occurring. However, the end result was calculated using only the best set of weights.

The resulting confusion matrix is as given below.



It performed pretty decently. The model was able to decently tell apart most of the emotions with anger being the most well-predicted one. However, it performed poorly for the sad emotion, and somewhat so for the fearful and neutral emotions.

### RNN Model (Transcription Analysis)

Knowing the audio clips were short, I decided to use GRU following the embedding layer, since there wasn't any need for greater memory for the model to function.

This is the architecture of the model.



My training/validation plot,



And my confusion matrix.



The very poor performance, almost close to a random guess (0.125) is due to one simple reason - the RAVDESS audio dataset didn't have emotions in the words themselves. For example, the phrase "kids are talking by the door" by itself doesn't have any discrimination for the emotion. The CNN model was able to classify the semantics only because the emotion was expressed acoustically, whereas the RNN model struggles due to the lack of linguistic cues.

I was tempted to try a transformer approach too as mentioned in the bonus task, but decided against it as I did not have much time for it. Also, it may have given diminishing returns as I already stated, the linguistics of the audio samples do not provide emotional cues, only the tones and pitch did like what the CNN was able to pick up on.

### CRNN Model (Multimodal)

I decided to go with path A for the multimodal approach. Expectations for this model were slim, since it was clear that the RNN model cannot provide much information about the emotion. I was hoping the CRNN model would atleast be able to relate the emotion of the audio with the position of the subjects in the text. In the end, it did not perform significantly better than the CNN model, granted it ran for a larger no. of epochs.

I decided against data augmentation here, unlike in my CNN, because I was cautious of the no. of text data multiplying 4 times. I feared it would cause the text model to interfere with the final results more.

This is the architecture of the model.



My training/validation plot,

And my confusion matrix. 

This model performed poorly in neutral, happy, and sad emotions. But was almost perfect for calm. It's difficult to comment the exact reasons behind this, and it may have been due to interference due to the poor-performing RNN model.

For all three models, I used the EarlyStopping callback, with restore_best_weights=True.

I obtained the following metrics across all three models. LR=5e-4, EPOCHS=300 to 600

Model name	              Accuracy (%)	      F1 Score
CNN Model	                55.56	              0.5460
RNN Model	                12.85	              0.0665
CRNN Model(Multi-model)	  56.94	              0.5344
(Note, all of this run using WSL2, to take advantage of GPU computing, which is how I managed to get a lot of epochs in.)

In conclusion, I believe with some further regularisation work, the CNN model could have out performed the CRNN model. It displayed a better spread of accuracies than the CRNN model.

