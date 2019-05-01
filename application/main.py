import librosa
from os import listdir
from os.path import isfile, join
import soundfile as sf
import numpy as np
import pandas as pd

import datetime
import sklearn

import tensorflow as tf
from tensorflow.keras import layers, models

path_2_training_samples = "application/urban-sound-classification/urban-sound-classification/train/Train/"

# get a list with all the audio filenames:
filenames = [file for file in listdir(path_2_training_samples)]


"""
example_file_path = join(path_2_examples1, filenames[0])
clip1, rate1 = librosa.load(example_file_path, sr=None)
spec_1 = librosa.feature.mfcc(y=clip1, sr=rate1)
import numpy as np
print(np.shape(spec_1))
"""

# read the labels:
path_2_csv = "application/urban-sound-classification/urban-sound-classification/train/train.csv"
labels = pd.read_csv(path_2_csv)['Class'].values

# map the string classes to integer classes:
integer_labels = []
classmap = {
    'air_conditioner': 0,
    'car_horn': 1,
    'children_playing': 2,
    'dog_bark': 3,
    'drilling': 4,
    'engine_idling': 5,
    'gun_shot': 6,
    'jackhammer': 7,
    'siren': 8,
    'street_music': 9
}
for label_index in range(0, len(labels)):
    labels[label_index] = classmap[labels[label_index]]

print("Read the audio data: " + str(datetime.datetime.now()))

# for testing purposes only ! erase or comment when ready !:
filenames = filenames[0:20]
labels = labels[0:20]

samples = []
for filename in filenames:
    x, sr = sf.read(join(path_2_training_samples, filename), always_2d=True)
    x = x.flatten('F')[:x.shape[0]]
    samples.append([np.mean(feature) for feature in librosa.feature.mfcc(x)])

print("Finished reading the audio data: " + str(datetime.datetime.now()))



# Even if there are some audio test samples, the test labels are missing, so we split the current data
# into 2 datasets: one for training and the other one for testing

training_audio, testing_audio, training_labels, testing_labels = sklearn.model_selection.train_test_split(samples, labels)
training_audio = np.array(training_audio)
training_labels = np.array(training_labels)
training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=10)


# build the model:
model = models.Sequential()
model.add(layers.Dense(30, input_dim=20))
model.add(layers.Dense(10, activation="sigmoid"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model:
model.fit(training_audio, training_labels, epochs=5)