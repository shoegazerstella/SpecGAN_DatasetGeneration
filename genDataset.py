#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import argparse
import librosa
from progressbar import ProgressBar



def prepare(y, sr=22050):
    y = librosa.to_mono(y)
    y = librosa.util.fix_length(y, sr)  # 1 second of audio
    y = librosa.util.normalize(y)
    return y


def get_fingerprint(y, sr=22050):
    y = prepare(y, sr)
    cqt = librosa.cqt(y, sr=sr, hop_length=2048)
    return cqt.flatten('F')


def normalize(x):
    x -= x.min(axis=0)
    x /= x.max(axis=0)
    return x


def basename(file):
    file = os.path.basename(file)
    return os.path.splitext(file)[0]


def genHarmPerc(inputDir, outputDir):
	print('- Generating harmonic and percussive separation')
	
	filelist=os.listdir(inputDir)

	for f in filelist:

		print('processing', f)

		# load audio
		y, sr = librosa.load(os.path.join(inputDir, f))

		# Compute the short-time Fourier transform of y
		D=librosa.stft(y)

		# decompose
		D_harmonic, D_percussive=librosa.decompose.hpss(D)

		# istft to convert back to time series
		n=len(y)

		y_out_harmonic=librosa.istft(D_harmonic, length=n)
		y_out_percussive=librosa.istft(D_percussive, length=n)

		harmonic_dir = os.path.join(outputDir, 'harmonic')
		percussive_dir = os.path.join(outputDir, 'percussive')

		try:
			os.makedirs(harmonic_dir)
		except:
		    pass

		try:
			os.makedirs(percussive_dir)
		except:
		    pass

		# save
		librosa.output.write_wav(os.path.join(harmonic_dir, f), y_out_harmonic, sr)
		librosa.output.write_wav(os.path.join(percussive_dir, f), y_out_percussive, sr)

	return harmonic_dir, percussive_dir

def splitSamples(sourceDir, outputDir, mode=None):

	print('- Splitting samples for dataset:', mode)

	source_filelist = os.listdir(sourceDir)

	for f in source_filelist:
	    
	    print('processing', f)

	    outputSamples = os.path.join(outputDir, 'dataset_' + mode)
	    
	    y, sr = librosa.load(os.path.join(sourceDir, f))
	    # trim silence at beginning and end
	    y, index = librosa.effects.trim(y)
	    # detect onsets
	    o_env = librosa.onset.onset_strength(y, sr=sr, feature=librosa.cqt)
	    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

	    vectors = []
	    words = []
	    filenames = []

	    onset_samples = list(librosa.frames_to_samples(onset_frames))
	    onset_samples = np.concatenate(onset_samples, len(y))
	    starts = onset_samples[0:-1]
	    stops = onset_samples[1:]

	    analysis_folder = sourceDir
	    samples_folder = os.path.join(outputSamples, f)
	    
	    try:
	        os.makedirs(samples_folder)
	    except:
	        pass

	    pbar = ProgressBar()
	    for i, (start, stop) in enumerate(pbar(zip(starts, stops))):
	        audio = y[start:stop]
	        filename = os.path.join(samples_folder, str(i) + '.wav')
	        librosa.output.write_wav(filename, audio, sr)
	        vector = get_fingerprint(audio, sr=sr)
	        word = basename(filename)
	        vectors.append(vector)
	        words.append(word)
	        filenames.append(filename)
	    np.savetxt(os.path.join(analysis_folder, 'vectors'), vectors, fmt='%.5f', delimiter='\t')
	    np.savetxt(os.path.join(analysis_folder, 'words'), words, fmt='%s')
	    np.savetxt(os.path.join(analysis_folder, 'filenames.txt'), filenames, fmt='%s')



if __name__ == '__main__':
	parser=argparse.ArgumentParser(description='Split audio in harmonic and percussive component, then split into multiple samples and save analysis.')
	parser.add_argument('-i', '--inputDir', type=str)
	parser.add_argument('-o', '--outputDir', type=str)
	args=parser.parse_args()

	inputDir=args.inputDir
	outputDir=args.outputDir

	# generate harmonic and percussive audio
	harmonic_dir, percussive_dir = genHarmPerc(inputDir, outputDir)

	# split samples based on onset
	splitSamples(harmonic_dir, outputDir, mode='harmonic')
	splitSamples(percussive_dir, outputDir, mode='percussive')