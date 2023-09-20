# Running the analysis:

The files ended with '_func.py' have no needs to be run.

## 01_eeg_preprocessing

* run_all_scripts.py (This preprocesses all datasets, including THINGS1, 

  THINGS2, Zhang & Wamsley)

* DEED.py (This file preprocesses one specific subject of DEED.)

* THINGS1.py (This file preprocesses one specific subject of THINGS1.)

* THINGS2.py (This file preprocesses one specific subject of THINGS2.)

* ZW.py (This file preprocesses one specific subject of Zhang & Wamsley.)

## 03_validation_test

* encoding.py (This file train and test the EEG encoding model)

## 04_dnn_fmaps

* alexnet_THINGS.py (This file extracts dnn alexnet feature maps from THINGS images)

* alexnet_SCIP.py (This file extracts dnn alexnet feature maps from SCIP images)