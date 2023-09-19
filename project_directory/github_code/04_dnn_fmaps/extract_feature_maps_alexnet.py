"""Extract and save the AlexNet feature maps of the training and test images,
and of the ILSVRC-2012 validation and test images.

Parameters
----------
pretrained : bool
	If True use a pretrained network, if False a randomly initialized one.
dataset : str
    Used dataset among 'THINGS_EEG2', 'SCIP'
project_dir : str
	Directory of the project folder.

"""

import argparse
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--dataset', default='THINGS_EEG2', type=str)
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

print('Extract feature maps AlexNet <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)


# =============================================================================
# Select the layers of interest and import the model
# =============================================================================
# Lists of AlexNet convolutional and fully connected layers
conv_layers = ['conv1', 'ReLU1', 'maxpool1', 'conv2', 'ReLU2', 'maxpool2',
	'conv3', 'ReLU3', 'conv4', 'ReLU4', 'conv5', 'ReLU5', 'maxpool5']
fully_connected_layers = ['Dropout6', 'fc6', 'ReLU6', 'Dropout7', 'fc7',
	'ReLU7', 'fc8']

class AlexNet(nn.Module):
	def __init__(self):
		"""Select the desired layers and create the model."""
		super(AlexNet, self).__init__()
		self.select_cov = ['maxpool1', 'maxpool2', 'ReLU3', 'ReLU4', 'maxpool5']
		self.select_fully_connected = ['ReLU6' , 'ReLU7', 'fc8']
		self.feat_list = self.select_cov + self.select_fully_connected
		self.alex_feats = models.alexnet(pretrained=args.pretrained).features
		self.alex_classifier = models.alexnet(pretrained=args.pretrained).classifier
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

	def forward(self, x):
		"""Extract the feature maps."""
		features = []
		for name, layer in self.alex_feats._modules.items():
			x = layer(x)
			if conv_layers[int(name)] in self.feat_list:
				features.append(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		for name, layer in self.alex_classifier._modules.items():
			x = layer(x)
			if fully_connected_layers[int(name)] in self.feat_list:
				features.append(x)
		return features

model = AlexNet()
if torch.cuda.is_available():
	model.cuda()
model.eval()


# =============================================================================
# Define the image preprocessing
# =============================================================================
centre_crop = trn.Compose([
	trn.Resize((224,224)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Load the images and extract the corresponding feature maps
# =============================================================================
# Extract the feature maps of 
# THINGS2 (1) training images, (2) test images.
# SCIP (1) cartoonflowers, (2) cartoonguitar, (3) cartoonpenguins.

# Image directories
if args.dataset == 'THINGS_EEG2':
	img_set_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
							'THINGS_EEG2', 'image_set')
elif args.dataset == 'SCIP':
	img_set_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
							'SCIP', 'stimuli' ,'image')
img_type = os.listdir(img_set_dir)
for p in img_type:
	part_dir = os.path.join(img_set_dir, p, 'selected')
	image_list = []
	for root, dirs, files in os.walk(part_dir):
		for file in files:
			if file.endswith(".jpg") or file.endswith(".JPEG") or file.endswith(".png"):
				image_list.append(os.path.join(root,file))
	image_list.sort()

	# Create the saving directory if not existing
	save_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', args.dataset, 
						 'dnn_feature_maps', 'full_feature_maps', 'alexnet', 
						 'pretrained-'+str(args.pretrained), p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	# Extract and save the feature maps
	for i, image in enumerate(tqdm(image_list)):
		img = Image.open(image).convert('RGB')
		input_img = V(centre_crop(img).unsqueeze(0))
		if torch.cuda.is_available():
			input_img=input_img.cuda()
		x = model.forward(input_img)
		feats = {}
		for f, feat in enumerate(x):
			feats[model.feat_list[f]] = feat.data.cpu().numpy()
		file_name = p + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), feats)