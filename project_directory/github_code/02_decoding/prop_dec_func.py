def feature_select(args, sub, img_type):
    import pandas as pd
    import numpy as np
    import os
    
    ### Load THINGS image data ###
    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS')

    # Load image names 
    img_dir  = os.path.join(THINGS_dir, 'image_set', img_type+'_images') 
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

    if img_type == 'training':
        posi_index = sorted_obj[-300:]
        nega_index = sorted_obj[:300]
    elif img_type == 'test':
        posi_index = sorted_obj[-50:]
        nega_index = sorted_obj[:50] 
    print(f'We select top / bottom, {len(posi_index)} / {len(nega_index)} images for {img_type}.')
  
    posi_index, nega_index = np.array(posi_index), np.array(nega_index)
        
    ### Load THINGS EEG data ###
    THINGS_prepr_dir = os.path.join(THINGS_dir, 
                                    'preprocessed_data', args.electrodes, 'sub-'+format(sub,'02'),
                                    'preprocessed_eeg_'+img_type+'.npy')
    THINGS_eeg_data = np.load(THINGS_prepr_dir, allow_pickle=True).item()
     
    # Notations
    repetitions = THINGS_eeg_data['preprocessed_eeg_data'].shape[1]
    channels = THINGS_eeg_data['ch_names']
    times = THINGS_eeg_data['times']
    
    ### Select features and targets ###
    X_posi, X_nega, y_posi, y_nega = [],[],[],[]
    
    # Appending training features and targets:
    if img_type == 'training':
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
    elif img_type =='test':
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
    import matplotlib.pyplot as plt
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
    plt.show()


def prop_dec(X, y, Xtest, ytest, args, times):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import permutation_test_score
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
    scores, perm_means, perm_stds = np.empty((num_times)), np.empty((num_times)), np.empty((num_times))
    perm_scores = np.empty((num_times, args.n_perm))
    
    ### Decoding and permutations ###
    for T_idx in tqdm(range(num_times), desc=f'{args.obj_prop} decoding training'):
        logreg = LogisticRegression().fit(X[:,:,T_idx], y)
        # decoding accuracy
        for c in range(num_conds):
            Ytest_pred_prob[c][T_idx] = logreg.predict_proba(Xtest[c,:,T_idx].reshape(1,-1))[0][1]
            Ytest_pred[c][T_idx] = logreg.predict(Xtest[c,:,T_idx].reshape(1,-1))
        binary_results[T_idx] = sum(Ytest_pred[:,T_idx] == ytest) / len(ytest)
        accuracy[:,T_idx] =  (Ytest_pred[:,T_idx] == ytest).astype(int)
        # permutation accuracies
        scores[T_idx], perm_scores[T_idx], _ = permutation_test_score(logreg, 
                                                          Xtest[:,:,T_idx], ytest, scoring="accuracy", 
                                                          cv=5, n_permutations=args.n_perm, n_jobs=-1)
        perm_means[T_idx] = np.mean(perm_scores[T_idx])
        perm_stds[T_idx] = np.std(perm_scores[T_idx])
        del logreg

    ### Plot the results ###
    fig, ax = plt.subplots(2)
    # Decoding result plot
    ax[0].set_title(f'{args.obj_prop} 2D decoding accuracy')
    im = ax[0].imshow(accuracy, cmap='viridis',extent=[-.2, .8, 0, 8000], 
            origin='lower', aspect='auto')
    cbar = plt.colorbar(im)
    cbar.set_label('Values')
    ax[0].set(xlabel = 'Time (s)', ylabel = "Images")
    # Permutation plot
    ax[1].set_title(f"Permutation scores vs Predicted scores of {args.obj_prop}")
    ax[1].set(xlabel = 'Time (s)', ylabel = "Accuracy")
    ax[1].plot([0, 0], [.45, .6], 'k--')
    ax[1].plot(times, perm_means, label = "Permutation scores", color="darkorange")
    ax[1].fill_between(times, perm_means - perm_stds, perm_means + perm_stds, alpha=0.2, color="darkorange")
    ax[1].plot(times, scores, label = "Predicted scores", color="navy")
    ax[1].legend(loc = 'best')
    fig.tight_layout()
    plt.show()

    max_time_idx = np.argmax(scores)
    print(f'The time with greatest score difference: {round(times[max_time_idx],2)}s')