"""Preprocess the  raw EEG data: channel selection, epoching, frequency
downsampling, baseline correction, multivariate noise normalization (MVNN),
sorting of the data image conditions and reshaping the data to:
Events × EEG channels × EEG time points.
Then, the data is saved.

Parameters
----------
sub : int
	Used subject.
n_ses : int
	Number of EEG sessions.
mvnn_dim : str
	Whether to compute the MVNN covariace matrices for each time point
	('time') or for each epoch/repetition ('epochs').
project_dir : str
	Directory of the project folder.

"""

import os
import argparse
from DEED_func import epoching
from DEED_func import mvnn
from DEED_func import save_prepr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sfreq', default=100, type=int)
# parser.add_argument('--mvnn_dim', default='time', type=str)
project_dir = os.path.join('D:/', 'UG', 'Research', 'Dream_Neural_Decoding',
                           'Shannon_Big_Big_Dream_Project', 'project_directory')
parser.add_argument('--project_dir', default=project_dir, type=str)
parser.add_argument('--PSG_number', default='295', type=str)
parser.add_argument('--chunk_size', default='100', type=int)
parser.add_argument('--mvnn_dim', default='time', type=str)
args = parser.parse_args()

print('>>> EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
    

# =============================================================================
# Epoch the data
# =============================================================================
# Epoching, baseline correction and frequency downsampling of the data.
# Then, the EEG data is reshaped to:
# (Epochs × EEG channels × EEG time points) x number of rems
# This step is applied independently to the dream data. 
epoched_data, ch_names, times = epoching(args)


# =============================================================================
# Multivariate Noise Normalization
# =============================================================================
# MVNN is applied independently to the data of each session.
whitened_data = mvnn(args, epoched_data)
del epoched_data


# =============================================================================
# Save the preprocessed data
# =============================================================================
# In this step the data of each rem has shape: 
# EEG channels × EEG time points
# Then, the preprocessed data is saved.
save_prepr(args, whitened_data, ch_names, times)

print(args.PSG_number, 'Done')