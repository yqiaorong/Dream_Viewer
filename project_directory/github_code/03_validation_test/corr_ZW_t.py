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
import argparse
import numpy as np
from tqdm import tqdm
import pingouin as pg
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from encoding_func import corr_t


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

# Test dreams EEG list
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
dreams_eegs_idx = [idx for idx, item in enumerate(dreams_eegs) if item in 
                   dreams_imgs]
print('The total number of dreams with feature maps: ', len(dreams_eegs_idx))

# The list of indices of dream images of target dream
dreams_imgs_idx = [idx for idx, item in enumerate(dreams_imgs) 
                   if item == dreams_eegs[dreams_eegs_idx[args.dream_idx]]]
print(f'The total number of images for dream {dreams_eegs[dreams_eegs_idx[args.dream_idx]]}: ',
      len(dreams_imgs_idx))


# =============================================================================
# Train the encoding model
# =============================================================================

### Load the training DNN feature maps ###
# Load the training DNN feature maps directory
dnn_train_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                'THINGS_EEG2', 'dnn_feature_maps', 'pca_feature_maps', 
                                args.dnn, 'pretrained-True', 'layers-all')
# Load the training DNN feature maps (16540, 3000)
dnn_fmaps_train = np.load(os.path.join(dnn_train_dir, 'pca_feature_maps_training.npy'), 
                            allow_pickle=True).item()

### Load the training EEG data ###
# Load the THINGS2 training EEG data directory
eeg_train_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                'THINGS_EEG2','preprocessed_data')
# Iterate over THINGS2 subjects
eeg_data_train = []
for train_subj in tqdm(range(1,11), desc='THINGS2 subjects'):
    # Load the THINGS2 training EEG data
    data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                  'preprocessed_eeg_training.npy'), allow_pickle=True).item()
    # Get the THINGS2 training channels and times
    if train_subj == 1:
        train_ch_names = data['ch_names']
    else:
        pass
    # Average the training EEG data across repetitions (16540,17,100)
    data = np.mean(data['preprocessed_eeg_data'], 1)
    # Crop the training EEG data between 0.1 and 0.25s (16540,17,15)
    data = data[:,:,30:45]
    # Remove the EEG data from 'POz' channel (16540,16,15)
    POz_idx = train_ch_names.index('POz')
    data = np.delete(data,POz_idx,axis=1)
    # Merge EEG channels and times (16540, 16 x 15)
    data = np.reshape(data, (data.shape[0],-1))
    # Append individual data
    eeg_data_train.append(data)
    del data
# Average the training EEG data across subjects : (16540,16 x 15)
eeg_data_train = np.mean(eeg_data_train,0)
# Delete unused channel names
del train_ch_names

### Train the encoding model ###
# Train the encoding models
reg = LinearRegression().fit(dnn_fmaps_train['all_layers'],eeg_data_train)


# =============================================================================
# Predict the EEG data of dreams
# =============================================================================

### Load the test dream DNN feature maps ###
# Load the test DNN feature maps directory
dnn_test_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                                args.test_dataset, 'dnn_feature_maps', 'pca_feature_maps', 
                                args.dnn, 'pretrained-True', 'layers-all')
# Load the test DNN feature maps (images, 3000)
dnn_fmaps_test = np.load(os.path.join(dnn_test_dir,'pca_feature_maps_dreams.npy'
                        ), allow_pickle=True).item()

### Predict the EEG test data using the encoding model ###
# Predict the test EEG data : (images,16 x 15)
pred_eeg_data_test = reg.predict(dnn_fmaps_test['all_layers'])
print('pred eeg data test shape: ', pred_eeg_data_test.shape)


# =============================================================================
# Compute the correlation scores for one dream
# =============================================================================	

# Set the cropped time points
crop_t = 1000
# Get the number of effective time points
eff_t = crop_t - 15
# The time points
times = np.linspace(-int(crop_t/100), -int(15/100), eff_t)

# The correlation scores (all images, times)
corr = []
# The correlation scores (all images)
mean_corr = []

# Iterate over images
for i in tqdm(range(len(dreams_imgs)), desc='correlation'):
    s, m = corr_t(args, pred_eeg_data_test, dreams_eegs_idx[args.dream_idx], i, crop_t)
    corr.append(s)
    mean_corr.append(m)
corr = np.array(corr)
mean_corr = np.array(mean_corr)

# Save the all correlation results of one dream to the dictionary
results = {}
results['corresponding_img_idx'] = dreams_imgs_idx
results['correlations'] = corr
results['mean_correlations'] = mean_corr
results['times'] = times

# Create the saving directory
save_dir = os.path.join(args.project_dir,'results',args.test_dataset,'correlation_scores_t')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
file_name = dreams_eegs[dreams_eegs_idx[args.dream_idx]]
np.save(os.path.join(save_dir, file_name), results)


# # =============================================================================
# # Compute the correlation mean scores for dreams
# # =============================================================================	

# # The array storing all correlation mean scores
# mcorr = np.empty((len(dreams_eegs_idx), len(dreams_imgs)))
# # Iterate over dreams
# for e, eeg_idx in enumerate(tqdm(dreams_eegs_idx)):
#     # Iterate over images
#     for i, img in enumerate(dreams_imgs):
#         _, mcorr[e,i] = corr_s(args, pred_eeg_data_test, eeg_idx, i, crop_t)    


# # =============================================================================
# # Plot the correlation scores matrix
# # =============================================================================	

# fig, ax = plt.subplots(1)
# ax.set_title(f'2D Correlation scores')
# im = ax.imshow(corr, cmap='viridis',
#                extent=[0, len(dreams_imgs), 0, len(dreams_eegs_idx)], 
#                origin='lower', aspect='auto')
# cbar = plt.colorbar(im)
# ax.set(xlabel = 'Images', ylabel = "Dreams")
# fig.tight_layout()
# plt.show()