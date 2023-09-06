import os
import argparse
import numpy as np
from scipy.stats import pearsonr as corr

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--electrodes', default='occipital', type=str)
args = parser.parse_args()

# Load the target THINGS EEG test data
THINGS_test_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS',
'preprocessed_data', args.electrodes, 'sub-'+format(args.sub,'02'),'preprocessed_eeg_test.npy')
test_data = np.load(THINGS_test_dir, allow_pickle=True).item()

times = test_data['times'] 
channels = test_data['ch_names'] 
test_data = test_data['preprocessed_eeg_data']

# Load all other THINGS EEG training data
sub_list = list(range(1,11))
sub_list.remove(args.sub)

training_data = []
for s in sub_list:
    THINGS_training_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS',
    'preprocessed_data', args.electrodes, 'sub-'+format(s,'02'),'preprocessed_eeg_training.npy')
    data = np.load(THINGS_training_dir, allow_pickle=True).item()
    data = data['preprocessed_eeg_data']
    training_data.append(data)
training_data = np.array(training_data)
print(training_data.shape)