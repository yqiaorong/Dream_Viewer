def epoching(args):
    """The function preprocesses the raw EEG file: channel selection, 
    creating annotations and events, re-reference, bandpass filter, epoching, 
    baseline correction and frequency downsampling. Then, it sorts the test 
    EEG data according to the image conditions.

    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    -------
    sort_data : list of float
        Epoched EEG test data.
    ch_names : list of str
        EEG channel names.
    times : float
        EEG time points.
    """

    import os
    import mne
    import numpy as np
    import pandas as pd
    
    ### Load the THINGS1 subject metadata ### 
    # Load the THINGS1 subject directory
    TH1_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',
                           'THINGS_EEG1','sub-'+format(args.subj,'02'))
    # Load the THINGS1 subject metadata
    dftsv = pd.read_csv(os.path.join(TH1_dir, 'sub-'+format(args.subj,'02')+
                                     '_task-rsvp_events.tsv'), delimiter='\t')
    
    ### Crop the THINGS1 subject metadata ###
    # Select the main 22248 images
    dftsv = dftsv[dftsv['teststimnumber'] == -1]
    # Select events relevant information
    dftsv = dftsv[['onset','object']] 
    
    ### Load the THINGS1 subject EEG data ###
    # Load the THINGS1 subject EEG directory
    TH1_EEG_dir = os.path.join(TH1_dir, 'sub-'+format(args.subj,'02')+
                               '_task-rsvp_eeg.vhdr')
    # Load the THINGS1 subject EEG raw file
    raw = mne.io.read_raw_brainvision(TH1_EEG_dir, preload=True)
    
    ### channel selection ###
    # Pick the main 64 channels
    if subj in [49, 50]:
        raw = raw.pick(raw.info['ch_names'][:63])
    else:
        pass
    # Select occipital channels
    chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
                                                   '^O *|^P *'))
    print('Picked occipital channels are:')
    print(chan_idx)
    new_chans = [raw.info['ch_names'][c] for c in chan_idx]
    print(new_chans)
    print('The total number of picked occipital channels: ', len(new_chans))
    # Pick occipital channels
    raw.pick(new_chans)
    
    ### Create annotations and events ###
    # Annotation onset
    onset = dftsv['onset'] # in seconds
    # Annotation duration
    duration = [0.05]*len(dftsv) # in seconds, too
    # Create annotations
    annot = mne.Annotations(onset=onset, duration=duration, 
                            description=['images']*len(dftsv))
    # Set annotations
    raw.set_annotations(my_annot)
    # Create events
    events, _ = mne.events_from_annotations(raw)
    
    ### Re-reference and bandpass filter all channels ###
    # Re-reference raw 'average'
    raw.set_eeg_reference()
    # Bandpass filtering
    raw.filter(l_freq=0.1, h_freq=100, n_jobs=-1)
    
    ### Epoching and resampling the raw data ###
    # Epoching
    epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0), 
                        preload=True)
    del raw
    # Resampling
    epochs.resample(args.sfreq)
    
    ### Get epoched channels and times ###
    ch_names = epochs.info['ch_names']
    times = epochs.times
    
    ### Sort epoched data according to the THINGS2 test images ###
    # Get epoched data
    epoched_data = epochs.get_data()
    del epochs
    # THINGS2 test images directory
    test_img_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',
                                 'THINGS_EEG2','image_set', 'test_images')
    # Create list of THINGS2 test images
    test_imgs = os.listdir(test_img_dir)
    # The sorted epoched data
    sort_data = []
    # Iterate over THINGS2 test images
    for test_img in test_imgs:
        # Get the indices of test image 
        indices = dftsv.index[dftsv['object'] == test_img[6:]]
        # Get the data of test image 
        data = [epoched_data[i, :, :] for i in indices]
        # Convert list to array
        data = np.array(data)
        # Average within the test image
        data = np.mean(data, axis=0)
        # Add the data to the test THINGS1 EEG data
        sort_data.append(data)
        del indices, data
    # Convert list to array
    sort_data = np.array(sort_data)
    
    return sort_data, ch_names, times