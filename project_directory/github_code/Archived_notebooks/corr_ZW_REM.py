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
import pandas as pd
from tqdm import tqdm
import pingouin as pg
from sklearn.linear_model import LinearRegression
from corr_func import corr_s


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='../project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
args = parser.parse_args()

print(f'>>> Test the encoding model on {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


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
    # Average the training EEG data across time (16540,17)
    data = np.mean(data,axis=2)
    # Remove the EEG data from 'POz' channel (16540,16)
    POz_idx = train_ch_names.index('POz')
    data = np.delete(data,POz_idx,axis=1)
    # Append individual data
    eeg_data_train.append(data)
    del data
# Average the training EEG data across subjects : (16540,16)
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
# Predict the test EEG data : (images,16)
pred_eeg_data_test = reg.predict(dnn_fmaps_test['all_layers'])
print('pred eeg data test shape: ', pred_eeg_data_test.shape)


# =============================================================================
# Load only REM sleeps
# =============================================================================	
df = pd.read_excel('../project_directory/results/Zhang_Wamsley/df.xlsx')

# Select only REMs
df_REM = df[df['sleep_stages'] == 5]
del df

# Select only REMs with images
df_REM_nonah = df_REM[df_REM['dreams_imgs'] != 'nah']
del df_REM

# Get REM dreams eeg indices
eeg_idx = np.unique(df_REM_nonah['dreams_idx'])

# Get REM dreams images indices
img_idx = np.unique(df_REM_nonah['dreams_imgs_idx']).astype(int)

# Only keep the eeg indices and images indices columns
idx_df = df_REM_nonah[['dreams_idx', 'dreams_imgs_idx']]
del df_REM_nonah
idx_df.to_excel(os.path.join(args.project_dir, 'results', 'Zhang_Wamsley', 
                         'REMs_df.xlsx'), index=False)

# Select predicted REMs EEG data
REM_pred_eeg = pred_eeg_data_test[img_idx]
del pred_eeg_data_test
print('REM pred eeg data test shape: ', REM_pred_eeg.shape)


# =============================================================================
# Compute the correlation scores for one dream
# =============================================================================	

for e in eeg_idx:

    # Test dreams EEG list
    ZW_EEG_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                            'Zhang_Wamsley', 'preprocessed_data')
    dreams_eegs_names = os.listdir(ZW_EEG_dir)
    
    # Get the indices of images for each REM dream
    single_dream_imgs_idx = list(idx_df[idx_df['dreams_idx'] == e]['dreams_imgs_idx'])
    print(single_dream_imgs_idx)

    # Set the cropped time points
    crop_t = 1000
    # The time points
    times = np.linspace(-int(crop_t/100), 0, crop_t)

    # The correlation scores (all images, times)
    corr = []
    # The correlation scores (all images)
    mean_corr = []

    # Iterate over images
    for i in tqdm(range(REM_pred_eeg.shape[0]), desc='correlation'):
        s, m = corr_s(args, REM_pred_eeg, e, i, crop_t)
        corr.append(s)
        mean_corr.append(m)
    corr = np.array(corr)
    mean_corr = np.array(mean_corr)

    # Save the all correlation results of one dream to the dictionary
    results = {}
    results['corresponding_img_idx'] = single_dream_imgs_idx
    results['correlations'] = corr
    results['mean_correlations'] = mean_corr
    results['times'] = times

    # Create the saving directory
    save_dir = os.path.join(args.project_dir,'results',args.test_dataset,'REMs',
                            'correlation_scores_s')
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)
    file_name = dreams_eegs_names[e][6:]
    np.save(os.path.join(save_dir, file_name), results)

    del single_dream_imgs_idx, corr, mean_corr, results