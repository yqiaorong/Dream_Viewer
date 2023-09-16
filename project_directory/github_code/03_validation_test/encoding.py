"""Test the THINGS2 decoding model on THINGS1

Parameters
----------
project_dir : str
	Directory of the project folder.
dnn_feature_maps : str
    The DNN feature maps used to train the encoding model.
test_subj : int
    Used test subject.
"""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--wake_data_dir',
					default='../project_directory/eeg_dataset/wake_data', 
					type=str)
parser.add_argument('--dnn_feature_maps',default='alexnet',type=str)
parser.add_argument('--test_subj', default=1, type=int)
args = parser.parse_args()

print('>>> Test the encoding model on THINGS1 <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# =============================================================================
# Load the DNN feature maps
# =============================================================================
# Load the training DNN feature maps directory
dnn_parent_dir = os.path.join(args.wake_data_dir, 'THINGS_EEG2', 'dnn_feature_maps',
    'pca_feature_maps', args.dnn_feature_maps, 'pretrained-True', 'layers-all')
# Load the training DNN feature maps
dnn_fmaps_train = np.load(os.path.join(dnn_parent_dir,'pca_feature_maps_training.npy'
                                       ), allow_pickle=True).item()
# Load the test DNN feature maps
dnn_fmaps_test = np.load(os.path.join(dnn_parent_dir,'pca_feature_maps_test.npy'
                                      ), allow_pickle=True).item()

# =============================================================================
# Train the encoding models
# =============================================================================
# Load the THINGS2 training EEG data directory
eeg_train_dir = os.path.join(args.wake_data_dir,'THINGS_EEG2','preprocessed_data')
# Iterate over THINGS2 subjects
eeg_data_train = []
for train_subj in range(1,11):
    # Load the THINGS2 training EEG data
    data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
						'preprocessed_eeg_training.npy'), allow_pickle=True).item()
    # Average the training EEG data across repetitions
    data_avg = np.mean(data['preprocessed_eeg_data'], 1)
    # Merge the dimension of EEG channels and times
    data_avg = np.reshape(data_avg,(data_avg.shape[0], -1))
    eeg_data_train.append(data_avg)
    if train_subj == 1:
          train_ch_names = data['ch_names']
          train_times = data['times']
    else:
        pass
    del data, data_avg
# Average the training EEG data across subjects
eeg_data_train = np.mean(eeg_data_train,0)
# Train the encoding models
reg = LinearRegression().fit(dnn_fmaps_train['all_layers'],eeg_data_train)
# Predict the THINGS1 test EEG data
pred_eeg_data_test = reg.predict(dnn_fmaps_test['all_layers'])
# Merge the dimension of EEG channels and times
pred_eeg_data_test = np.reshape(pred_eeg_data_test,
                                (-1,len(train_ch_names),len(train_times)))

# =============================================================================
# Test the encoding models
# =============================================================================

# Load the THINGS1 test EEG data 
eeg_test_dir = os.path.join(args.wake_data_dir,'THINGS_EEG1','preprocessed_data',
                            'sub-'+format(args.test_subj,'02'))
eeg_data_test = np.load(os.path.join(eeg_test_dir, 'preprocessed_eeg_test.npy'),
                        allow_pickle=True).item()
test_ch_names = eeg_data_test ['ch_names']
test_times = eeg_data_test ['times']

# Average the test EEG data across repetitions
eeg_data_test_avg = np.mean(eeg_data_test['preprocessed_eeg_data'], 1)

# Calculate the encoding accuracy
encoding_accuracy = np.zeros((len(test_ch_names),len(test_times)))
for t in range(len(test_times)):
    for c in range(len(test_ch_names)):
        encoding_accuracy[c,t] = corr(pred_eeg_data_test[:,c,t],
            eeg_data_test_avg[:,c,t])[0]
        
# Plot the results
plt.figure()
plt.plot([-.2, .8], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
plt.plot(eeg_data_test['times'], np.mean(encoding_accuracy, 0), label='alexnet')
plt.xlabel('Time (s)')
plt.xlim(left=-.2, right=.8)
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-.1, top=.7)
plt.title('Encoding accuracy on THINGS1')
plt.legend(loc='best')
plt.show()