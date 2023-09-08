def category_raw(args):
    """
    function for identifying which high-level categories are present in a text, and 
    High-level categories are taken from the THINGS+ dataset (generated by WordNet).
    """
    import os
    import spacy
    import tqdm
    import mne
    import pandas as pd
    from collections import Counter
    from pathlib import Path

    # The filepath of the categories file (should never change).
    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS')
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

def epoching(args, raw):
    import mne
    import numpy as np
    from sklearn.utils import shuffle

    ### Drop unused channels ###
    chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
        '^O *|^P *'))
    new_chans = [raw.info['ch_names'][c] for c in chan_idx]
    raw.pick(new_chans)

    ### Baseline correction and resampling ###
    # Resampling
    raw = raw.resample(args.sfreq)

    ch_names = raw.info['ch_names']
    times = raw.times
    epoched_data = raw.get_data()    

    ### Output ###
    return epoched_data, ch_names, times

def mvnn(args, epoched_data):
    import numpy as np
    from sklearn.discriminant_analysis import _cov
    import scipy
    
    ### Compute the covariance matrices with shape: EEG channels × EEG channels  ###
    sigma = np.empty((epoched_data.shape[0],  epoched_data.shape[0]))
    
    # Compute covariace matrices at each time point, and then
    # average across time points
    if args.mvnn_dim == "time":
        sigma = _cov(epoched_data.T, shrinkage='auto')

    # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5) 
    # The matrix power is -0.5, which represents inverse square root

    ### Whiten the data ###
    whitened_data = np.reshape((epoched_data.swapaxes(0, 1) @ sigma_inv).swapaxes(0, 1), 
                               (epoched_data.shape[0], epoched_data.shape[1]))
    ### Output ###
    return whitened_data

def save_prepr(args, whitened_data, ch_names, times, ID_select):
    import numpy as np
    import os

    ### Save the data ###
    save_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 'Zhang_Wamsley', 'preprocessed_data')
    # Create the directory if it doesn't exist #
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Insert the data into a dictionary #
    my_dict = {
        'preprocessed_eeg_data': whitened_data,
        'ch_names': ch_names,
        'times': times
    }

    folder_name = args.category
    save_folder = os.path.join(save_dir, folder_name)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    file_name = 'prepr_'+ID_select
    np.save(os.path.join(save_folder, file_name), my_dict)
    del my_dict