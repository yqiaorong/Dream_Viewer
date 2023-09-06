def epoching(args):
    """This function first converts the raw EEG data to MNE raw format, and
    performs epoching, frequency downsampling, baseline correction.

    Parameters
    ----------
    args : Namespace
        Input arguments.
        
    Returns
    -------
    epoched_data : list of array, [array(1,6,times,) × number of rems]
        Epoched EEG data. 
    ch_names : list of str
        EEG channel names.
    times : list of array, [array(times,) × number of rems]
        EEG time points.
        
    """
    
    import os
    import mne
    import numpy as np
    import scipy
    import pandas as pd   
    import datetime 
    
    DEED_dir = os.path.join('eeg_dataset', 'dream_data', 'DEED', 'raw_data') 
    eeg_dir = os.path.join(DEED_dir, 'raw_mat')        
    
    ### Load the metadata and eeg data ###
    f1 = pd.read_excel(os.path.join(args.project_dir, DEED_dir,'Status_identification_of_each_stage_of_EEG.xlsx'))
    f2 = pd.read_excel(os.path.join(args.project_dir, DEED_dir,'Emotional_ratings_excel_filesfiles.xlsx'))
    
    if args.PSG_number in f2['PSG_number'].values:
        raw_data = scipy.io.loadmat(os.path.join(args.project_dir, eeg_dir, args.PSG_number))
    
        ### Convert to MNE raw format ###
        eeg_data = raw_data["Data"]*10**-7 # rescale
        del raw_data
        ch_names=['F3','F4','FT7','FT8','T7','T8','RF']
        sfreq=200
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, 
                        ch_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg'])
        raw = mne.io.RawArray(eeg_data, info)
        raw.set_eeg_reference(ref_channels=['RF'])
        raw.drop_channels(ch_names='RF')
        del eeg_data 
    
        # Set measurement time #
        for index, row in f1.iterrows():
            psg_number = row['PSG_number']
            if psg_number == args.PSG_number:
                date = str(f1.loc[index, 'Date'])
                exp_onset = f1.loc[index, 'Start_recording'] 
                break
                print(date)
        measurement_time = pd.Timestamp(f'{date[0:4]}-{date[4:6]}-{date[6:8]} {exp_onset}')
        raw.set_meas_date(measurement_time.timestamp())
    
        ### Create events ###
        # Create dream onsets and ends timestamps #
        dream_onsets = []
        dream_ends = []
        f2['PSG_number'] = f2['PSG_number'].ffill()
        for index, row in f2.iterrows():
            psg_number = str(row['PSG_number'])
            if psg_number == args.PSG_number:
                if not isinstance(f2.loc[index, "REM_period_start_time"], datetime.time):
                    pass
                else: 
                    dream_onset = pd.Timestamp(
                        f'{date[0:4]}-{date[4:6]}-{date[6:8]} {f2.loc[index, "REM_period_start_time"]}') + pd.Timedelta(days=1)
                    dream_end = pd.Timestamp(
                        f'{date[0:4]}-{date[4:6]}-{date[6:8]} {f2.loc[index, "End_time_of_REM_period"]}') + pd.Timedelta(days=1)
                    dream_onsets.append(dream_onset)
                    dream_ends.append(dream_end)
        dream_onset_sec = [(i - measurement_time).total_seconds() for i in dream_onsets]
        dream_dur_sec = [(f - i).total_seconds() for i, f in zip(dream_onsets,dream_ends)]
        print(dream_dur_sec)
        
        # Create events #
        events = [[int(x), 0, 1] for x in dream_onset_sec]
        print(events[0])
        
        ### Create epochs and get outputs ###
        epoched_data = []
        times = []
        for i, tmax in enumerate(dream_dur_sec):
            epoch = mne.Epochs(raw, np.reshape(events[i],(1,3)), tmin=0, 
                            tmax=tmax, baseline=(0,0), preload=True) 
        
            # Resampling
            if args.sfreq < 200:
                epoch.resample(args.sfreq)

            # Ouput #
            epoched_data.append(epoch.get_data())
            ch_names = epoch.info['ch_names']
            times.append(epoch.times)
            del epoch
        del raw
        
    else:
        epoched_data = []
        ch_names = []
        times = []
        print(args.PSG_number, ':discarded')
    
    return epoched_data, ch_names, times


def mvnn(args, epoched_data):
    """
    Compute the covariance matrices of the EEG data (calculated for each
    time-point of each emotion labels). The inverse of the
    resulting covariance matrix is used to whiten the EEG data.
    
    Parameters
    ----------
    epoched_data : list of array, [array(1,6,times,) × number of rems]
        Epoched EEG data. 
    
    Returns
    -------
    whitened_data : list of array, [array(6,times,) × number of rems]
        Whitened EEG data. 

    """
    
    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy
    
    ### Compute the covariance matrices ###
    sigma = np.empty((len(epoched_data), epoched_data[0].shape[1],  epoched_data[0].shape[1]))
    
    # Break each epoch to more epochs each with duration 1.0s
    for index, epoch in enumerate(epoched_data):
        sub = [epoch[:, :, i:i+args.chunk_size] for i in range(0, epoch.shape[2], args.chunk_size)]
        data = np.concatenate(sub, axis=0)

        # Epoch covariance matrix of shape: EEG channels × EEG channels  
        
        # Compute covariace matrices at each time point, and then
        # average across time points
        if args.mvnn_dim == "time":
            sigma[index] = np.mean([_cov(data[:,:,t], shrinkage='auto') for t in range(data.shape[2])], axis=0)
            
        # Compute covariace matrices at each epoch,
        # and then average across epochs
        elif args.mvnn_dim == "epochs":
            sigma[index] = np.mean([_cov(np.transpose(data[e]),
                shrinkage='auto') for e in range(data.shape[0])], axis=0)
            
    # Average the covariance matrices across image partitions
    sigma_tot = sigma.mean(axis=0)

    # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5) 
    # The matrix power is -0.5, which represents inverse square root

    ### Whiten the data ###
    whitened_data = []
    for epoch in epoched_data: 
        whitened_data.append(np.reshape((epoch.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2), (epoch.shape[1], epoch.shape[2])))

    ### Output ###
    return whitened_data


def save_prepr(args, whitened_data, ch_names, times):
    """The preprocessed data  of each dream has format:
    EEG channels × EEG time points.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    whitened_data : list of array, [array(6,times,) × number of rems]
        Whitened EEG data. 
    ch_names : list of str
        EEG channel names.
    times : list of array, [array(times,) × number of rems]
        EEG time points.

    """
    import numpy as np
    import os
    ### Save the data ###
    save_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 'DEED', 'preprocessed_data')
    # Create the directory if it doesn't exist #
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for index, (data, t) in enumerate(zip(whitened_data, times)):
        # Insert the data into a dictionary #
        my_dict = {
            'preprocessed_eeg_data': data,
            'ch_names': ch_names,
            'times': t
        }
        folder_name = args.PSG_number[5:8]
        save_folder = os.path.join(save_dir, folder_name)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        file_name = folder_name + '_' + str(index+1)
        np.save(os.path.join(save_folder, file_name), my_dict)
        del my_dict