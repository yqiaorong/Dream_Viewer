# Dream_Decoding

Using waking EEG dataset to decodes dream content. 

## Datasets

The open datasets currently used are：

* [THINGS EEG1](https://www.nature.com/articles/s41597-021-01102-7) 

* [THINGS EEG2 dataset](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)

* [SCIP dataset](https://www.nature.com/articles/s41597-023-02287-9)

* [Zhang & Wamsley dataset](https://onlinelibrary.wiley.com/doi/10.1111/psyp.13368)

The path of download:

1. [THINGS EEG1](https://openneuro.org/datasets/ds003825/versions/1.2.0): ../project_directory/eeg_dataset/wake_data/THINGS_EEG1/

2. THINGS EEG2: 
   
   * [Raw EEG data](https://osf.io/crxs4/): ../project_directory/eeg_dataset/wake_data/THINGS_EEG2/raw_data/

   * [Image set](https://osf.io/y63gw/): ../project_directory/eeg_dataset/wake_data/THINGS_EEG2/image_set/

2. THINGS images meta data: 

   * [category](https://osf.io/vehr3): ../project_directory/eeg_dataset/wake_data/THINGS_EEG2/

   * [object properties ratings](https://osf.io/3kwn2): ../project_directory/eeg_dataset/wake_data/THINGS_EEG2/
   
3. [DEED](http://www.deeddataset.com/#/download):

   * EEG data for the whole night: ../project_directory/eeg_dataset/dream_data/DEED/raw_data/

   * Emotional ratings: ../project_directory/eeg_dataset/dream_data/DEED/raw_data/

   * Status identification: ../project_directory/eeg_dataset/dream_data/DEED/raw_data/

4. [SCIP](https://openneuro.org/datasets/ds004306/versions/1.0.1): ../project_directory/eeg_dataset/wake_data/SCIP/

5. [Zhang & Wamsley 2019](https://doi.org/10.6084/m9.figshare.22226692): ../project_directory/eeg_dataset/dream_data/Zhang_Wamsley

## The analysis

The files ended with '_func.py' have no needs to be run.

### 01_eeg_preprocessing

* run_all_scripts.py (This preprocesses all datasets, including THINGS1, 

  THINGS2, Zhang & Wamsley)

* DEED.py (This file preprocesses one specific subject of DEED.)

* THINGS1.py (This file preprocesses one specific subject of THINGS1.)

* THINGS2.py (This file preprocesses one specific subject of THINGS2.)

* ZW.py (This file preprocesses one specific subject of Zhang & Wamsley.)

### 03_validation_test

* encoding.py (This file train and test the EEG encoding model)

### 04_dnn_fmaps

* alexnet_THINGS.py (This file extracts dnn alexnet feature maps from THINGS images)

* alexnet_SCIP.py (This file extracts dnn alexnet feature maps from SCIP images)