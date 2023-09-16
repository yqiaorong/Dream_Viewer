"""
Preprocesses the raw EEG file: channel selection, creating annotations 
and events, re-reference, bandpass filter, epoching, baseline correction,
frequency downsampling, sorting the test EEG data according to the image 
conditions, multivariate noise normalization (MVNN) and reshaping the 
data to: Image conditions  × EEG channels × EEG time points.
Then, the test EEG data is saved.

Parameters

----------
project_dir : str
	Directory of the project folder.
subj : int
	Used subject.
sfreq : int
	Downsampling frequency.
mvnn_dim : str
	Whether to compute the MVNN covariace matrices for each time point
	('time') or for each epoch/repetition ('epochs').


"""

import argparse
from THINGS1_func import epoching, mvnn, save_prepr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--subj', default=1, type=int)
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--mvnn_dim', default='time', type=str)
args = parser.parse_args()

print('')
print('>>> EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220


# =============================================================================
# Epoch and sort the data
# =============================================================================
epoched_test, ch_names, times = epoching(args)
epoched_train, _, _ = epoching(args)


# =============================================================================
# Multivariate Noise Normalization
# =============================================================================
whitened_test, whitened_train = mvnn(args, epoched_test, epoched_train)
del epoched_test, epoched_train


# =============================================================================
# Merge and save the preprocessed data
# =============================================================================
# In this step the data of all sessions is merged into the shape:
# Image conditions × EGG repetitions × EEG channels × EEG time points
# Then, the preprocessed data of the test and training data partitions is saved.
save_prepr(args, whitened_test, whitened_train, img_conditions_train, ch_names,
	times, seed)