"""PCA is performed on the DNN feature maps to reduce their dimensionality.
PCA is applied on either the feature maps of single DNN layers, or on the
appended feature maps of all layers.

Parameters
----------
project_dir : str
	Directory of the project folder.
dataset : str
    Used dataset (THINGS_EEG2), or ('SCIP').
dnn : str
	Used DNN among 'alexnet', 'resnet50', 'cornet_s', 'moco'.
pretrained : bool
	If True use the pretrained network feature maps, if False use the randomly
	initialized network feature maps.
layers : str
	Whether to use 'all' or 'single' layers.
n_components : int
	Number of DNN feature maps PCA components retained.
"""

import argparse
import numpy as np
import os

from pca_func import train_scaler_pca, apply_scaler_pca

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--dataset', default='THINGS_EEG2', type=str)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--layers', default='single', type=str)
parser.add_argument('--n_components', default=1000, type=int)

args = parser.parse_args()

print(f'>>> Apply PCA on the {args.dataset} feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# =============================================================================
# Apply PCA on the training images feature maps
# =============================================================================
# The standardization and PCA statistics computed on the THINGS training images.
# feature maps are also applied to the test images feature maps.

scaler, pca, fmaps_train, all_layers, layer_names = train_scaler_pca(args)

# Save the downsampled feature maps
save_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data','THINGS_EEG2',
						'dnn_feature_maps','pca_feature_maps', args.dnn, 
						'pretrained-'+str(args.pretrained), 'layers-'+args.layers)
file_name = 'pca_feature_maps_training_images'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_train)
del fmaps_train

# =============================================================================
# Apply PCA on the test images feature maps
# =============================================================================
if args.dataset == 'THINGS_EEG2':
    apply_scaler_pca(args, 'test_images', scaler, pca, all_layers, layer_names)
if args.dataset == 'SCIP':
    img_categories = ['cartoonflowers', 'cartoonguitar', 'cartoonpenguins']
    for img_category in img_categories:
        apply_scaler_pca(args, img_category, scaler, pca, all_layers, layer_names)