def feature_select(args):
    import pandas as pd
    import numpy as np
    import os
    
    ### Load THINGS image data ###
    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS')

    # Load image names 
    img_dir  = os.path.join(THINGS_dir, 'image_set', args.img_type+'_images') 
    concepts_dir = os.listdir(img_dir)
    concepts = [concept[6:] for concept in concepts_dir]
    
    # load image property metadata
    img_meta_dir = os.path.join(THINGS_dir, 'objectProperties_meanRatings.tsv')
    img_data = pd.read_csv(img_meta_dir, delimiter='\t')

    img_data = img_data[img_data['uniqueID'].isin(concepts)]
    img_data = img_data.reset_index()

    ### Select object property ###
    obj = list(img_data.loc[:, args.obj_prop].items())
    sorted_obj = sorted(obj, key=lambda x: x[1]) # Ascending order

    if args.img_type == 'training':
        posi_index = sorted_obj[-300:]
        nega_index = sorted_obj[:300]
    elif args.img_type == 'test':
        posi_index = sorted_obj[-50:]
        nega_index = sorted_obj[:50] 
    print(f'We select top / bottom, {len(posi_index)} / {len(nega_index)} images for {args.img_type}.')
  
    posi_index, nega_index = np.array(posi_index), np.array(nega_index)
            
    if args.img_type == 'training':
        No_p = posi_index.shape[0]*10
        No_n = nega_index.shape[0]*10
    elif args.img_type =='test':
        No_p = posi_index.shape[0]
        No_n = nega_index.shape[0]
        
    ### Load THINGS EEG data ###
    THINGS_prepr_dir = os.path.join(THINGS_dir, 
                                    'preprocessed_data', args.electrodes, 'sub-'+format(sub,'02'),
                                    'preprocessed_eeg_'+args.img_type+'.npy')
    THINGS_eeg_data = np.load(THINGS_prepr_dir, allow_pickle=True).item()
     
    # Notations
    repetitions = THINGS_eeg_data['preprocessed_eeg_data'].shape[1]
    channels = THINGS_eeg_data['preprocessed_eeg_data'].shape[2]
    times = THINGS_eeg_data['preprocessed_eeg_data'].shape[3]
    
    ### Select features and targets ###
    X_posi, X_nega, y_posi, y_nega = [],[],[],[]
    
    # Appending training features and targets:
    if args.img_type == 'training':
        for p in posi_index[:,0]: 
            for x in range(int(p*10),int(p*10+10)):
                for r in range(repetitions):
                    X_posi.append(THINGS_eeg_data['preprocessed_eeg_data'][x,r,:,:])
                    y_posi.append(1)
        for n in nega_index[:,0]: 
            for x in range(int(n*10),int(n*10+10)):
                for r in range(repetitions):
                    X_nega.append(THINGS_eeg_data['preprocessed_eeg_data'][x,r,:,:])
                    y_nega.append(0)               
    # Appending test features and targets:
    elif args.img_type =='test':
        for p in posi_index[:,0]: 
            for r in range(repetitions):
                X_posi.append(THINGS_eeg_data['preprocessed_eeg_data'][int(p),r,:,:])
                y_posi.append(1)
        for n in nega_index[:,0]: 
            for r in range(repetitions):
                X_nega.append(THINGS_eeg_data['preprocessed_eeg_data'][int(n),r,:,:])
                y_nega.append(0)
        
    # Combine both positive and negative features
    X_posi, X_nega, y_posi, y_nega = np.array(X_posi), np.array(X_nega), np.array(y_posi), np.array(y_nega)
    X = np.concatenate((X_posi, X_nega))
    y = np.concatenate((y_posi, y_nega))
        
    return X, y, channels, times


def LearnCurve(X,y,args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LearningCurveDisplay
    import numpy as np

    logreg = LogisticRegression().fit(X[:,:,args.rand_t_idx], y)

    common_params = {
    "X": X[:,:,args.rand_t_idx],
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": 5,
    "score_type": "both",
    "n_jobs": -1,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",}
    LearningCurveDisplay.from_estimator(logreg, **common_params)


def prop_dec(X, y, Xtest, ytest, args):
    from sklearn.linear_model import LogisticRegression
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np

    # Notations
    num_conds =  Xtest.shape[0]
    num_channels = Xtest.shape[1]
    num_times = Xtest.shape[2]
    
    # Results
    Ytest_pred_prob = np.empty((num_conds, num_times))
    Ytest_pred = np.empty((num_conds, num_times))
    binary_results = np.empty((num_times))
    accuracy = np.empty((num_conds, num_times))
    
    # Decoding
    for T_idx in tqdm(range(num_times), desc=f'{args.obj_prop} decoding training'):
        logreg = LogisticRegression().fit(X[:,:,T_idx], y)
        for c in range(num_conds):
            Ytest_pred_prob[c][T_idx] = logreg.predict_proba(Xtest[c,:,T_idx].reshape(1,-1))[0][1]
            Ytest_pred[c][T_idx] = logreg.predict(Xtest[c,:,T_idx].reshape(1,-1))
        binary_results[T_idx] = sum(Ytest_pred[:,T_idx] == ytest) / len(ytest)
        accuracy[:,T_idx] =  (Ytest_pred[:,T_idx] == ytest).astype(int)
        del logreg

    # Plot the decoding results
    plt.figure(figsize=(8, 6))
    plt.imshow(accuracy, cmap='viridis',extent=[-.2, .8, 0, 8000], 
            origin='lower', aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Values')
    plt.xlabel('time(s)')
    plt.ylabel('images')
    plt.title(f'{args.obj_prop} 2D decoding accuracy')
    plt.show()

    return plt