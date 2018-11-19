from __future__ import print_function
import librosa
import pickle
import csv

filenames = ['../audio_files/CV_stimuli/a/a_dg_CV/a_dg01_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg02_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg03_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg04_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg05_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg06_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg07_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg08_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg09_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg10_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg11_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg12_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg13_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg14_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg15_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg16_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg17_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg18_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg19_CV.wav',
    '../audio_files/CV_stimuli/a/a_dg_CV/a_dg20_CV.wav'
]

new_filenames = ['../Spectrograms/a_dg01_CV.csv',
    '../Spectrograms/a_dg02_CV.csv',
    '../Spectrograms/a_dg03_CV.csv',
    '../Spectrograms/a_dg04_CV.csv',
    '../Spectrograms/a_dg05_CV.csv',
    '../Spectrograms/a_dg06_CV.csv',
    '../Spectrograms/a_dg07_CV.csv',
    '../Spectrograms/a_dg08_CV.csv',
    '../Spectrograms/a_dg09_CV.csv',
    '../Spectrograms/a_dg10_CV.csv',
    '../Spectrograms/a_dg11_CV.csv',
    '../Spectrograms/a_dg12_CV.csv',
    '../Spectrograms/a_dg13_CV.csv',
    '../Spectrograms/a_dg14_CV.csv',
    '../Spectrograms/a_dg15_CV.csv',
    '../Spectrograms/a_dg16_CV.csv',
    '../Spectrograms/a_dg17_CV.csv',
    '../Spectrograms/a_dg18_CV.csv',
    '../Spectrograms/a_dg19_CV.csv',
    '../Spectrograms/a_dg20_CV.csv'
]


def flatten(l):
    return [item for sublist in l for item in sublist]

for i,curr_file in enumerate(filenames):
    y, sr = librosa.load(curr_file)
    x = librosa.feature.melspectrogram(y=y, sr=sr)
    with open(new_filenames[i], "wb") as f:
        writer = csv.writer(f)
        writer.writerows(x)





