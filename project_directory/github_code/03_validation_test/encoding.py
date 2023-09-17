"""Test the THINGS2 decoding model on THINGS1

Parameters
----------
project_dir : str
	Directory of the project folder.
dnn_feature_maps : str
    The DNN feature maps used to train the encoding model.
"""

import argparse
import numpy as np
import pingouin as pg
from tqdm import tqdm
from matplotlib import pyplot as plt

from encoding_func import train_model, test_model

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--wake_data_dir',
					default='../project_directory/eeg_dataset/wake_data', 
					type=str)
parser.add_argument('--dnn_feature_maps',default='alexnet',type=str)
args = parser.parse_args()

print('>>> Test the encoding model on THINGS1 <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# =============================================================================
# Train the encoding model and predict the EEG test data
# =============================================================================
pred_eeg = train_model(args)

# =============================================================================
# Test the encoding model on each test subject
# =============================================================================

# Test subjects list
test_subjs = [x for x in range(1, 51) if x != 6]

# Get the encoding accuracy for each subject
tot_accuracy = np.empty((49,100))
for i, test_subj in enumerate(tqdm(test_subjs)):
    accuracy, times = test_model(args, test_subj, pred_eeg)
    tot_accuracy[i] = accuracy
        
# Plot all the results
plt.figure(1)
plt.plot([-.2, .8], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
for i in range(49):
    plt.plot(times, tot_accuracy[i], alpha=0.2)
plt.plot(times, np.mean(tot_accuracy,0), color='k', label='Correlation mean score')
plt.xlabel('Time (s)')
plt.xlim(left=-.2, right=.8)
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-.1, top=.3)
plt.title(f'Encoding accuracy on THINGS1 ({args.dnn_feature_maps})')
plt.legend(loc='best')

# Set random seed for reproducible results
seed = 20200220
# Set the confidence interval
ci = np.empty((2,len(times)))

plt.figure(2)
plt.plot([-.2, .8], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
# Calculate the confidence interval
for i in range(len(times)):
    ci[:,i] = pg.compute_bootci(tot_accuracy[:,i], func='mean', seed=seed)
# Plot the results with confidence interval
plt.plot(times, np.mean(tot_accuracy,0), color='salmon', label='Confidence interval')
plt.fill_between(times, np.mean(tot_accuracy,0), ci[0], color='salmon', alpha=0.2)
plt.fill_between(times, np.mean(tot_accuracy,0), ci[1], color='salmon', alpha=0.2)
plt.xlabel('Time (s)')
plt.xlim(left=-.2, right=.8)
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-.1, top=.3)
plt.title(f'Encoding accuracy on THINGS1 ({args.dnn_feature_maps})')
plt.legend(loc='best')

plt.show()