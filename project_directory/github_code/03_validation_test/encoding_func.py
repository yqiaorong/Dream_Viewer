def train_model(args):
    """The function trains the encoding model using LogisticRegression. X train 
    is THINGS2 dnn feature maps and Y train is the THINGS2 real EEG training 
    data
    
    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    ----------
    reg : Logistic regression 
        The trained encoding model.
    """

    import os
    import pickle
    import numpy as np
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression

    ### Load the training DNN feature maps ###
    # Load the training DNN feature maps directory
    dnn_parent_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                  'THINGS_EEG2', 'dnn_feature_maps', 'pca_feature_maps', 
                                  args.dnn, 'pretrained-True', 'layers-all')
    # The list of training feature maps
    dnn = os.listdir(dnn_parent_dir)
    # Load the training DNN feature maps
    for d in dnn:
        if d.endswith('training.npy'):
            dnn_fmaps_train = np.load(os.path.join(dnn_parent_dir,
                                                   'pca_feature_maps_training.npy'), 
                                                   allow_pickle=True).item()
        elif d.endswith('training'):
            with open(os.path.join(dnn_parent_dir,'pca_feature_maps_training'),'rb') as f:
                dnn_fmaps_train = pickle.load(f)
    
    ### Train the encoding models ###
    # Load the THINGS2 training EEG data directory
    eeg_train_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                 'THINGS_EEG2','preprocessed_data')
    # Iterate over THINGS2 subjects
    eeg_data_train = []
    for train_subj in tqdm(range(1,11), desc='THINGS2 subjects'):
        # Load the THINGS2 training EEG data
        data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                            'preprocessed_eeg_training.npy'), allow_pickle=True).item()
        # Average the training EEG data across repetitions
        data_avg = np.mean(data['preprocessed_eeg_data'], 1)
        # Merge the dimension of EEG channels and times
        data_avg = np.reshape(data_avg,(data_avg.shape[0], -1))
        eeg_data_train.append(data_avg)
        if train_subj == 1:
            train_ch_names = data['ch_names']
        else:
            pass
        del data, data_avg
    # Average the training EEG data across subjects
    eeg_data_train = np.mean(eeg_data_train,0)
    print(eeg_data_train.shape)
    if args.test_dataset == 'Zhang_wamsley':
        # Remove the EEG data from 'Oz' channel
        eeg_data_train = np.delete(eeg_data_train,12,axis=1)
        # Remove the channel name 
        train_ch_names.pop[12]
    else: 
        pass
    # Train the encoding models
    reg = LinearRegression().fit(dnn_fmaps_train['all_layers'],eeg_data_train)

    ### Output ###
    return reg

def test_model_THINGS1(args, reg, test_subj):
    """The function tests the encoding model by correlating the predicted EEG 
    test data with real EEG test data.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    reg : Logistic regression 
        The trained encoding model.
    test_subj : int
        Used test subject.
    
    Returns
    ----------
    encoding_accuracy: list of float
        The encoding accuracy / correlation result.
    test_times : list of float
        EEG time points.
    """

    import os
    import numpy as np
    from scipy.stats import pearsonr as corr

    ### Test the encoding model ###
    # Load the THINGS1 test EEG data 
    eeg_test_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data',
                                    'THINGS_EEG1', 'preprocessed_data', 
                                    'sub-'+format(test_subj,'02'))
    eeg_data_test = np.load(os.path.join(eeg_test_dir, 'preprocessed_eeg_test.npy'),
                            allow_pickle=True).item()
    # Get test channel names and times
    test_ch_names = eeg_data_test ['ch_names']
    test_times = eeg_data_test ['times']
    # Average the test EEG data across repetitions
    eeg_data_test_avg = np.mean(eeg_data_test['preprocessed_eeg_data'], 1)

    ### Load the test DNN feature maps ###
    # Load the test DNN feature maps directory
    dnn_parent_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 
                                      'THINGS_EEG2', 'dnn_feature_maps', 'pca_feature_maps', 
                                      args.dnn, 'pretrained-True', 'layers-all')
    # Load the test DNN feature maps
    dnn_fmaps_test = np.load(os.path.join(dnn_parent_dir,'pca_feature_maps_test.npy'
                                        ), allow_pickle=True).item()

    ### Predict the EEG test data using the encoding model ###
    # Predict the test EEG data
    pred_eeg_data_test = reg.predict(dnn_fmaps_test['all_layers'])
    # Merge the dimension of EEG channels and times
    pred_eeg_data_test = np.reshape(pred_eeg_data_test,
                                    (-1,len(test_ch_names),len(test_times)))
    
    ### Test the encoding model ###
    # Calculate the encoding accuracy
    encoding_accuracy = np.zeros((len(test_ch_names),len(test_times)))
    for t in range(len(test_times)):
        for c in range(len(test_ch_names)):
            encoding_accuracy[c,t] = corr(pred_eeg_data_test[:,c,t],
                eeg_data_test_avg[:,c,t])[0]
    # Average the encoding accuracy across channels
    encoding_accuracy = np.mean(encoding_accuracy,0)
            
    ### Output ###
    return encoding_accuracy, test_times