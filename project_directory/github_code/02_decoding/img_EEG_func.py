def concepts_select(args, img_type):
    import os
    import pandas as pd

    ### Load the image meta data with higher level categories ###
    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS')
    img_meta_dir = os.path.join(THINGS_dir, 'things_concepts.tsv')
    img_meta_data = pd.read_csv(img_meta_dir, delimiter='\t')
    
    ### First select concepts under one category ###
    # Notice: I choose 'Top-down Category (WordNet)' as reference for categorization
    df = img_meta_data[img_meta_data['Top-down Category (WordNet)'] == args.category]
    init_select = []
    init_select.extend(df['uniqueID'].tolist())
    print(f'The total number of concepts under category {args.category}: ', len(init_select))

    ### Secondly select concepts only appeared in training/test set ###
    img_dir  = os.path.join(THINGS_dir, 'image_set', img_type+'_images') 
    concepts = os.listdir(img_dir)
    
    final_select = []
    for i_s in init_select:
        for c in concepts:
            if i_s == c[6:]:
                final_select.append(c)
    print(f'The number of concepts under category {args.category} in {img_type} set: ', len(final_select))

    return final_select

def img_EEG_match(args, concepts, sub, img_type):
    import os
    import numpy as np
    
    ### Load THINGS prepr data ###
    THINGS_prepr_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS',
    'preprocessed_data', 'occipital', 'sub-'+format(sub,'02'),'preprocessed_eeg_'+img_type+'.npy')
    data = np.load(THINGS_prepr_dir, allow_pickle=True).item()
    data = data['preprocessed_eeg_data']
    
    ### Select data with the right category ###
    idx = [int(c[:5])-1 for c in concepts]
    EEG_select = []
    for i in idx:
        EEG_select.append(data[i:i+10,:,:,:])
    EEG_select = np.array(EEG_select)
    data = EEG_select.reshape(EEG_select.shape[0]*EEG_select.shape[1], 
                              EEG_select.shape[2], EEG_select.shape[3], EEG_select.shape[4])

    return data