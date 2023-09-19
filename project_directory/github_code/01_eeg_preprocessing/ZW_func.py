def category_raw(args):
    """
    function for identifying which high-level categories are present in a text, and 
    High-level categories are taken from the THINGS+ dataset (generated by WordNet).
    """
    import os
    import spacy
    import tqdm
    import pandas as pd
    from collections import Counter

    # The filepath of the categories file (should never change).
    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS_EEG2')
    import_path_categories = os.path.join(THINGS_dir, "category53_longFormat.tsv")

    # The filepath of the dream reports file.
    Zhang_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 'Zhang_Wamsley')
    import_path_reports = os.path.join(Zhang_dir, 'Data', 'Reports.csv')

    # The column name of the dream reports (should be present in the dream reports file).
    reports_column = "Text of Report"

    # Load both files.
    df_categories = pd.read_csv(import_path_categories, sep="\t")
    df_reports = pd.read_csv(import_path_reports)

    # Extract a dictionary that maps words to categories.
    word2category_mapping = df_categories.set_index("Word")["category"].to_dict()

    # Extract the dream reports as a list.
    reports = df_reports[reports_column].tolist()

    # Load the spaCy model.
    nlp = spacy.load("en_core_web_sm")

    # Create an empty list to hold the results.
    results = []
    # Loop over each dream report.
    for report in tqdm.tqdm(reports):
        # Tokenize and lemmatize the dream report (i.e., split into words and get root word form).
        doc = nlp(report)
        lemmas = [token.lemma_ for token in doc]
        # Get word categories for each word that belongs to one.
        lemma_categories = [word2category_mapping.get(lemma, "none") for lemma in lemmas]
        # Get counts of each word category.
        category_counts = Counter(lemma_categories)
        # Append this result to the results list.
        results.append(category_counts)

    # Convert the results to a DataFrame.
    df = pd.Series(results).apply(pd.Series).fillna(0).astype(int)

    # The list of the indices of dream reports under the same dream category
    report_idx = []

    for index, row in df.iterrows():
        if row[args.category] != 0:
            report_idx.append(index)

    # The list of 'subject ID' under the same dream category
    IDs = df_reports.loc[report_idx, 'Case ID'].tolist()
    IDs = [f"{'0' + num if len(num) == 2 else num}_{text}" for num, text in [ID.split('_') for ID in IDs]]
    print(f'The number of dreams under category {args.category}: {len(IDs)}')
    
    return IDs

def epoching(args):
    """The function preprocesses the raw dream EEG file: channel selection, 
    re-reference, bandpass filter, epoching, and frequency downsampling. 

    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    -------
    epoched_data : array of shape (channel,time)
        Epoched EEG test data.
    ch_names : list of str
        EEG channel names.
    times : list of float
        EEG time points.
    """

    import os
    import mne

    ### Load dream EEG data
    # The path of target PSG file
    Zhang_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
    'Zhang_Wamsley', 'Data', 'PSG')
    # Read the target PSG file
    raw = mne.io.read_raw_edf(os.path.join(Zhang_dir, 'subject'+args.PSG+'.edf'),
    preload=True)

    ### Channel selection of dream dataset ###
    # The dream dataset misses the 'Poz' channel, idx 12 in THINGS2
    # THINGS2 channel names
    TH_ch = ['Pz','P3','P7','O1','Oz','O2','P4','P8','P1','P5','Po7','Po3','Po4','Po8','P6','P2']
    # Modify THINGS2 channel names according to ZW
    TH_ch = [ch+'-REF' for ch in TH_ch]
    # Pick up occipital and parietal channels
    raw.pick(TH_ch)

    ### Re-reference and bandpass filter all channels ###
    # Re-reference raw 'average'
    raw.set_eeg_reference()  
    # Bandpass filter
    raw.filter(l_freq=0.1, h_freq=100)
    
    ### Resampling and epoching ###
    # Downsampling
    raw = raw.resample(args.sfreq)
    # Get epoched data
    epoched_data = raw.get_data()  

    ### Get epoched channels and times ###
    ch_names = raw.info['ch_names']
    times = raw.times
    
    ### Output ###
    return epoched_data, ch_names, times

def mvnn(epoched_data):
    """Compute the covariance matrices of the EEG data (calculated for each
    time-point). The inverse of the resulting averaged covariance matrix is 
    used to whiten the EEG data.

    Parameters
    ----------
    epoched_data : array of shape (image,repetition,channel,time)
        Epoched EEG data.

    Returns
    -------
    whitened_data : array of shape (image,repetition,channel,time)
        Whitened EEG data.
    """

    import numpy as np
    from sklearn.discriminant_analysis import _cov
    import scipy
    
    ### Compute the covariance matrices with shape: EEG channels × EEG channels  ###
    
    # Compute covariace matrices at each time point, and then
    # average across time points
    sigma = _cov(epoched_data.T, shrinkage='auto')

    # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5) 
    # The matrix power is -0.5, which represents inverse square root

    ### Whiten the data ###
    whitened_data = (epoched_data.swapaxes(0, 1) @ sigma_inv).swapaxes(0, 1)
    ### Output ###
    return whitened_data

def save_prepr(args, whitened_data, ch_names, times):
    """The preprocessed dream EEG data is saved

    Parameters
    ----------
    args : Namespace
        Input arguments.
    whitened_data : array of shape (image,repetition,channel,time)
        Whitened EEG data.
    ch_names : list of str
        EEG channel names.
    times : list of float
        EEG time points.

    """
    import numpy as np
    import os

    ### Save the data ###
    save_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
    'Zhang_Wamsley', 'preprocessed_data')
    # Create the directory if it doesn't exist 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # Insert the data into a dictionary 
    my_dict = {
        'preprocessed_eeg_data': whitened_data,
        'ch_names': ch_names,
        'times': times
    }
    # Save the preprocessed EEG data
    np.save(os.path.join(save_dir, 'prepr_'+args.PSG), my_dict)
    del my_dict