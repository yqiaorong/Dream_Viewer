"""Test the THINGS2 decoding model on Zhang & Wamsley

Parameters
----------
train_data_dir : str
	Directory of the training data folder.
dnn_feature_maps : str
    The DNN feature maps used to train the encoding model.
test_dataset : str
    Used test dataset ('Zhang_Wamsley')
"""

import os
import sys
import argparse
import numpy as np
import pingouin as pg
from matplotlib import pyplot as plt

from encoding_func import train_model_spatial, test_model_ZW_spatial, plot_single

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',
					default='../project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--dream_idx',default=0,type=int)
args = parser.parse_args()

print(f'>>> Test the encoding model on {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
     

# =============================================================================
# Match the EEG data with images
# =============================================================================

# Test dreams eeg list
ZW_EEG_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'preprocessed_data')
dreams_eegs = os.listdir(ZW_EEG_dir)
dreams_eegs = [s[6:-4].replace('_', '') for s in dreams_eegs]

# Test dreams images list
ZW_img_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'images')
dreams_imgs = os.listdir(ZW_img_dir)
dreams_imgs = [s[6:].split('_')[0] for s in dreams_imgs]

# The list of indices of dreams with feature maps
dreams_eegs = [item for item in dreams_eegs if item in dreams_imgs]

# Select the dream imgs satisfying the chosen dream
selected_imgs_idx = []
for i, dream_img in enumerate(dreams_imgs):
	if dream_img == dreams_eegs[args.dream_idx]:
		selected_imgs_idx.append(i)
print('The number of dream images:', len(selected_imgs_idx))


# =============================================================================
# Train the encoding model and predict the EEG test data
# =============================================================================

# The predicted eeg : array with shape 
# (num_dreams x num_imgs each dream, channels x time points)
if len(selected_imgs_idx) != 0:
    pred_eeg = train_model_spatial(args)
else:
	sys.exit()


# =============================================================================
# Test the encoding model on Zhang & Wamsley each dream
# =============================================================================

# Compute the correlation results    
correlations = test_model_ZW_spatial(args, pred_eeg, selected_imgs_idx)

        
# =============================================================================
# Plot the correlation results
# =============================================================================

# plt.figure(1, figsize=(10,4))
# plt.plot([-21, 1], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
# # Set the plot colour spectum
# cmap = "cividis"
# colours = plt.colormaps[cmap](np.linspace(0,1,8))
# # Plot

# for i in range(8):
#     plt.plot(times, correlations[i], color = colours[i], alpha=0.2)
# plt.plot(times, np.mean(correlations,0), color='k', lw = 0.5, label='Correlation mean score')
# plt.xlabel('Time (s)')
# plt.xlim(left=-21, right=1)
# plt.ylabel('Pearson\'s $r$')
# plt.ylim(bottom=-1, top=1)
# plt.title(f'Encoding accuracy of all 8 images')
# plt.legend(loc='best')

times = np.linspace(-20, 0, 2000)
for i in range(len(selected_imgs_idx)):
    _ = plot_single(args, times, correlations, i, dreams_eegs[args.dream_idx])

# =============================================================================
# Plot the average correlation results with confidence intervals
# =============================================================================

# # Set random seed for reproducible results
# seed = 20200220

# fig = plt.figure(figsize=(12, 6))
# gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[3, 1])

# ### Temporal correlation results ###
# ax1 = plt.subplot(gs[0, 0])
# ax1.plot([-21, 1], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
# # Set the confidence interval
# ci = np.empty((2,2000))
# # Calculate the confidence interval
# for i in range(2000):
#     ci[:,i] = pg.compute_bootci(correlations[:,i], func='mean', seed=seed)
# # Plot the results with confidence interval
# ax1.plot(times, np.mean(correlations,0), color='salmon', label='Confidence interval')
# ax1.fill_between(times, np.mean(correlations,0), ci[0], color='salmon', alpha=0.2)
# ax1.fill_between(times, np.mean(correlations,0), ci[1], color='salmon', alpha=0.2)
# ax1.set(xlabel='Time (s)', ylabel='Pearson\'s $r$')
# ax1.set_xlim(left=-21, right=1)
# ax1.set_ylim(bottom=-1, top=1)
# ax1.set_title(f'Encoding accuracy with Confidence interval of {dreams_eegs[dream_idx]}')

# ### Histogram of correlation results
# ax2 = plt.subplot(gs[0, 1])
# # Plot the upper limit
# ax2.hist(np.mean(correlations,0)+ci[1], bins=20, color='violet', edgecolor='black')
# # Plot the mean histogram
# ax2.hist(np.mean(correlations,0), bins=20, color='lightskyblue', edgecolor='black')
# # Plot the lower limit
# ax2.hist(np.mean(correlations,0)-ci[0], bins=20, color='mediumspringgreen', edgecolor='black')
# ax2.set_title(f'Histogram of correlation results of {dreams_eegs[dream_idx]}')
# ax2.set_xlim(-1,1)
# ax2.set(xlabel='Pearson\'s $r$', ylabel='frequency')

# plt.legend(loc='best')

# ### Plot each correlation result ###
# fig, ax = plt.subplots(4, 2, figsize=(10, 4))
# # Decoding result plot
# for i in range(4):
#     ax[i][0].plot(times, correlations[i], label = f"image {i}", color=colours[i])
#     ax[i][0].legend(loc = 'best')
# for i in range(4):
#     ax[i][1].plot(times, correlations[int(i+4)], label = f"image {int(i+4)}", color=colours[int(i+4)])  
#     ax[i][1].legend(loc = 'best')

# ax[3][0].set(xlabel = 'Time (s)')
# ax[3][1].set(xlabel = 'Time (s)')
# fig.suptitle(f"Temporal correlation scores of {dreams_eegs[dream_idx]}")
# fig.tight_layout()

# plt.show()