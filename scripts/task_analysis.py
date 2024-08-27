import classification_algorithms as ca
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, PredefinedSplit, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


def train_validate_and_test_intersubject(df, remove_tasks=[], remove_from='both'):
    features = ['ax_mean', 'ax_var', 'ay_mean', 'ay_var', 'az_mean', 'az_var',
                'a2_mean', 'a2_var', 'a2_min', 'a2_max', 'entropy']
    target = ['t_mid']
    classifier = RandomForestClassifier(class_weight='balanced')
    param_grid = {'n_estimators': [100, 200, 500, 1000, 1200, 1500]}
    new_subs = []
    sublist = [(sub, subdata[~(subdata.task.isin(remove_tasks))]) for sub, subdata in df.groupby('subject') if subdata.task.isin(remove_tasks).any()] \
        if (remove_from == 'test' or remove_from == 'both') else df.groupby('subject')

    for sub, subdata in sublist:
        # the subject sub is taken as the test set
        traindata = df[df['subject'] != sub]

        # remove tasks while training
        if (remove_from == 'train' or remove_from == 'both'):
            traindata = traindata[~(traindata.task.isin(remove_tasks))]

        x_train = traindata[features].values
        x_test = subdata[features].values

        y_train = traindata[target].values.reshape(-1)
        y_test = subdata[target].values.reshape(-1)

        # tune hyperparameters by cross validation
        split = ca.split_intersubject(traindata)
        grid_search = GridSearchCV(classifier, param_grid, cv=split)

        grid_search.fit(x_train, y_train)

        best_classifier = grid_search.best_estimator_

        # run the classifier on the test set
        subdata['pred'] = best_classifier.predict(x_test)
        new_subs.append(subdata)

    return pd.concat(new_subs)


def train_validate_test_intra_task_combinations(df, remove_tasks=[]):
    classifier = RandomForestClassifier(class_weight='balanced')
    param_grid = {'n_estimators': [100, 200, 500, 1000, 1200, 1500]}

    features = ['ax_mean', 'ax_var', 'ay_mean', 'ay_var', 'az_mean', 'az_var',
                'a2_mean', 'a2_var', 'a2_min', 'a2_max', 'entropy']
    target = ['t_mid']

    col = features + ['time', 'subject', 'task']
    grp1, grp2, grp3, grp4 = [], [], [], []
    sublist = [(sub, subdata) for sub, subdata in df.groupby('subject') if
               subdata.task.isin(remove_tasks).any()] if len(remove_tasks) != 0 else df.groupby('subject')

    for i in range(10):
        for sub, subdata in sublist:
            traindata = pd.DataFrame()
            testdata = pd.DataFrame()
            subdata = subdata.reset_index()

            x_train, x_test, y_train, y_test = train_test_split(subdata[col].values,
                                                                subdata[target].values.reshape(-1),
                                                                test_size=0.2,
                                                                stratify=subdata[target].values.reshape(-1))

            traindata[col] = x_train
            testdata[col] = x_test
            traindata[target] = y_train.reshape((-1, 1))
            testdata[target] = y_test.reshape((-1, 1))

            # tune hyperparameters by cross validation for FULL TRAINSET
            split = ca.split_intrasubject(traindata, features, target, 4)
            grid_search = GridSearchCV(classifier, param_grid, cv=split)

            grid_search.fit(traindata[features].values, traindata[target].values.reshape(-1))

            best_classifier = grid_search.best_estimator_

            # TRAINSET WITH TASK REMOVED
            taskrem_train = traindata[~(traindata.task.isin(remove_tasks))]

            # tune hyperparameters by cross validation for TASK REMOVED TRAINSET
            try:
                split = ca.split_intrasubject(taskrem_train, features, target, 4)
            except:
                break
            grid_search2 = GridSearchCV(classifier, param_grid, cv=split)

            grid_search2.fit(taskrem_train[features].values, taskrem_train[target].values.reshape(-1))

            best_classifier2 = grid_search2.best_estimator_

            # run the classifier on the FULL TESTSET
            grp1_test = testdata
            grp1_test['pred'] = best_classifier.predict(grp1_test[features].values)
            grp1.append(grp1_test.assign(iteration=i))

            # TESTSET WITH TASK REMOVED
            taskrem_test = testdata[~(testdata.task.isin(remove_tasks))]

            # run the classifier on the TASK REMOVED TESTSET
            grp2_test = taskrem_test
            grp2_test['pred'] = best_classifier.predict(grp2_test[features].values)
            grp2.append(grp2_test.assign(iteration=i))

            # run the classifier on the FULL TESTSET
            grp3_test = testdata
            grp3_test['pred'] = best_classifier2.predict(grp3_test[features].values)
            grp3.append(grp3_test.assign(iteration=i))

            # run the classifier on the TASK REMOVED TESTSET
            grp4_test = taskrem_test
            grp4_test['pred'] = best_classifier2.predict(grp4_test[features].values)
            grp4.append(grp4_test.assign(iteration=i))

    grp1 = pd.concat(grp1).set_index('time')
    grp2 = pd.concat(grp2).set_index('time')
    grp3 = pd.concat(grp3).set_index('time')
    grp4 = pd.concat(grp4).set_index('time')

    return grp1, grp2, grp3, grp4


def generate_intrasubject_output(hand='right', task='walk'):
    if hand == 'right':
        subject_type = 'control'
        _, data = ca.read_data(subject_type)
    elif hand == 'left':
        subject_type = 'control'
        data, _ = ca.read_data(subject_type)
    elif hand == 'affected':
        subject_type = 'patient'
        data, _ = ca.read_data(subject_type)
    elif hand == 'unaffected':
        subject_type = 'patient'
        _, data = ca.read_data(subject_type)
    else:
        raise Exception(f"Invalid hand: {hand}. Use 'left', 'right', 'affected' or 'unaffected' instead.")

    if task == 'walk':
        remove_tasks = ['Walk25', 'OnSwitch', 'OpenDoor']
    elif task == 'openbottle':
        remove_tasks = ['OpenBottle']
    elif task == 'drinkcup':
        remove_tasks = ['DrinkCup']
    else:
        raise Exception(f"Invalid task: {task}. Use 'walk', 'openbottle' or 'drinkcup' instead.")

    feat = ca.compute_features(data[(data['task'] != ' ')], 4)
    tgt = 'l' if (hand == 'left' or hand == 'affected') else 'r'

    rfl1, rfl2, rfl3, rfl4 = train_validate_test_intra_task_combinations(feat, remove_tasks=remove_tasks)
    rfl1 = rfl1.rename(columns={'pred': tgt})[[tgt, 'iteration']]
    rfl2 = rfl2.rename(columns={'pred': tgt})[[tgt, 'iteration']]
    rfl3 = rfl3.rename(columns={'pred': tgt})[[tgt, 'iteration']]
    rfl4 = rfl4.rename(columns={'pred': tgt})[[tgt, 'iteration']]

    rfl1.to_csv(subject_type + f'/classifier outputs/rf{tgt}_intra_{task}.csv')
    rfl2.to_csv(subject_type + f'/classifier outputs/rf{tgt}_intra_test_without_{task}.csv')
    rfl3.to_csv(subject_type + f'/classifier outputs/rf{tgt}_intra_train_without_{task}.csv')
    rfl4.to_csv(subject_type + f'/classifier outputs/rf{tgt}_intra_no_{task}.csv')


def generate_intersubject_output(hand='right', task='walk'):
    if hand == 'right':
        subject_type = 'control'
        _, data = ca.read_data(subject_type)
    elif hand == 'left':
        subject_type = 'control'
        data, _ = ca.read_data(subject_type)
    elif hand == 'affected':
        subject_type = 'patient'
        data, _ = ca.read_data(subject_type)
    elif hand == 'unaffected':
        subject_type = 'patient'
        _, data = ca.read_data(subject_type)
    else:
        raise Exception(f"Invalid hand: {hand}. Use 'left', 'right', 'affected' or 'unaffected' instead.")

    if task == 'walk':
        remove_tasks = ['Walk25', 'OnSwitch', 'OpenDoor']
    elif task == 'openbottle':
        remove_tasks = ['OpenBottle']
    elif task == 'drinkcup':
        remove_tasks = ['DrinkCup']
    else:
        raise Exception(f"Invalid task: {task}. Use 'walk', 'openbottle' or 'drinkcup' instead.")

    feat = ca.compute_features(data[(data['task'] != ' ')], 4)
    tgt = 'l' if (hand == 'left' or hand == 'affected') else 'r'

    rfl1 = train_validate_and_test_intersubject(feat, remove_tasks=remove_tasks, remove_from=None).rename(
        columns={'pred': tgt})[[tgt]]
    rfl2 = train_validate_and_test_intersubject(feat, remove_tasks=remove_tasks, remove_from='test').rename(columns={'pred': tgt})[[tgt]]
    rfl3 = train_validate_and_test_intersubject(feat, remove_tasks=remove_tasks, remove_from='train').rename(columns={'pred': tgt})[[tgt]]
    rfl4 = train_validate_and_test_intersubject(feat, remove_tasks=remove_tasks, remove_from='both').rename(columns={'pred': tgt})[[tgt]]

    rfl1.to_csv(subject_type + f'/classifier outputs/rf{tgt}_inter_{task}.csv')
    rfl2.to_csv(subject_type + f'/classifier outputs/rf{tgt}_inter_test_without_{task}.csv')
    rfl3.to_csv(subject_type + f'/classifier outputs/rf{tgt}_inter_train_without_{task}.csv')
    rfl4.to_csv(subject_type + f'/classifier outputs/rf{tgt}_inter_no_{task}.csv')


def plot_results_for_task(task, subject_type, inter=True, subject=None):
    if inter:
        files = ['rfl_inter_' + task + '.csv', 'rfl_inter_train_without_' + task + '.csv',
                 'rfl_inter_test_without_' + task + '.csv', 'rfl_inter_no_' + task + '.csv', 
                 'rfr_inter_' + task + '.csv', 'rfr_inter_train_without_' + task + '.csv',
                 'rfr_inter_test_without_' + task + '.csv', 'rfr_inter_no_' + task + '.csv']
        methods_dict = ca.get_outputs(subject_type, names=files)
        dfs = []
        datas = ca.read_data(subject_type)
        for hand, data in zip(['left', 'right'], datas):
            subresults = []
            for name, df in methods_dict.items():
                if name.find('rf' + hand[0]) != -1:
                    name = '_'.join(name.split('_')[2:])
                    temp = pd.merge(data[['gnd', 'subject']], df, on='time')
                    for sub, subdata in temp.groupby('subject'):
                        subresults += [ca.confmatrix(subdata[hand[0]], subdata['gnd']).assign(subject=sub, method=name)]
            table = pd.concat(subresults)
            dfs.append(table)

    else:
        files = ['rfl_intra_' + task + '.csv', 'rfl_intra_train_without_' + task + '.csv',
                 'rfl_intra_test_without_' + task + '.csv', 'rfl_intra_no_' + task + '.csv', 
                 'rfr_intra_' + task + '.csv', 'rfr_intra_train_without_' + task + '.csv',
                 'rfr_intra_test_without_' + task + '.csv', 'rfr_intra_no_' + task + '.csv']

        methods_dict = ca.get_outputs(subject_type, names=files)
        dfs = []
        datas = ca.read_data(subject_type)
        for hand, data in zip(['left', 'right'], datas):
            subresults = []
            for name, df in methods_dict.items():
                if name.find('rf' + hand[0]) != -1:
                    name = '_'.join(name.split('_')[2:])
                    temp = pd.merge(data[['gnd', 'subject']], df, on='time')
                    for sub, subdata in temp.groupby('subject'):
                        for i, itdf in subdata.groupby('iteration'):
                            subresults += [
                                ca.confmatrix(itdf[hand[0]], itdf['gnd']).assign(subject=sub, method=name, iteration=i)]

            table = pd.concat(subresults)
            dfs.append(table)
    plt.figure(figsize=(5, 5))
    df = dfs[0] if not subject else dfs[0].loc[dfs[0].subject == subject].copy()
    df = df.fillna(0)
    ax = plt.subplot(121)
    title = 'left' if subject_type == 'control' else 'affected'
    plt.title(title, fontsize=15)

    df['group'] = [r'$\bar{t_r}\bar{t_e}$' if x.find('no') != -1 else r'$t_r\bar{t_e}$' if x.find(
        'test') != -1 else r'$\overline{t_r}t_e$' if x.find('train') != -1 else r'$t_rt_e$' for x in df.method]

    sns.pointplot(data=df, x='group', y='sensitivity', join=False, ci='sd', capsize=0.1)
    sns.pointplot(data=df, x='group', y='specificity', join=False, color='tab:red', ci='sd', capsize=0.1)
    plt.ylim(-0.05, 1.05)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=19)
    plt.xlabel(None)
    plt.ylabel(None)

    ax = plt.subplot(122)
    title = 'right' if subject_type == 'control' else 'unaffected'
    plt.title(title, fontsize=15)
    df = dfs[1] if not subject else dfs[1].loc[dfs[1].subject == subject].copy()
    df = df.fillna(0)

    df['group'] = [r'$\bar{t_r}\bar{t_e}$' if x.find('no') != -1 else r'$t_r\bar{t_e}$' if x.find(
        'test') != -1 else r'$\bar{t_r}t_e$' if x.find('train') != -1 else r'$t_rt_e$' for x in df.method]
    sns.pointplot(data=df, x='group', y='sensitivity', join=False, ci='sd', capsize=0.1)
    sns.pointplot(data=df, x='group', y='specificity', join=False, color='tab:red', ci='sd', capsize=0.1)
    plt.ylim(-0.05, 1.05)

    labels = [r'sensitivity', r'specificity']
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none", linewidth=10)[0]
    colors = ['tab:blue', 'tab:red']
    handles = [f("o", colors[i]) for i in range(2)]

    ax.set_facecolor('gainsboro')
    ax.set_alpha(0.7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([])
    plt.xticks(fontsize=19)

    plt.xlabel(None)
    plt.ylabel(None)

    plt.legend(handles, labels, bbox_to_anchor=(1, -0.1), loc='upper right', fontsize=17, ncol=2)
    cv = 'inter' if inter else 'intra'
    plt.suptitle('-'.join([subject_type, task, cv]), fontsize=18)
    return dfs
    
def plot_proportion_vs_performance():
    plt.figure(figsize=(7, 10))
    aff, _ = ca.read_data('patient')
    files = ['rf_intra.csv', 'rf_intra_ax.csv', 'rf_intra_mean.csv', 'rf_intra_var.csv']
    subresults = []
    for name in files:
        df = pd.read_csv('patient/classifier outputs/' + name, parse_dates=['time'], index_col='time')
        temp = pd.merge(aff[['gnd', 'subject']], df, on='time')
        for sub, subdata in temp.groupby('subject'):
            for i, itdf in subdata.groupby('iteration'):
                subresults += [ca.confmatrix(itdf['l'], itdf['gnd']).assign(subject=sub, method=name, iteration=i)]
    intra_aff = pd.concat(subresults)
    df = intra_aff.groupby(['method', 'subject']).mean()
    df['percentage of functional use'] = df['true positive']+df['false negative']
    plt.subplot(211)
    sns.scatterplot(data=df, x='percentage of functional use', y='sensitivity', hue='method', s=70)
    sns.regplot(data=df, x='percentage of functional use', y='sensitivity', color='k', scatter=False)
    plt.legend([],[], frameon=False)
    plt.xlabel('Percentage of Functional Use', fontsize=18)
    plt.ylabel('Sensitivity', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.subplot(212)
    sns.scatterplot(data=df, x='percentage of functional use', y='specificity', hue='method', legend=False, s=70)
    sns.regplot(data=df, x='percentage of functional use', y='specificity', color='k', scatter=False)
    plt.xlabel('Percentage of Functional Use', fontsize=18)
    plt.ylabel('Specificity', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none", linewidth=10)[0]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = ['full', 'mean of ax', 'all mean', 'all mean and variance']
    handles = [f("o", colors[i]) for i in range(len(labels))]
    
    plt.legend(handles, labels, bbox_to_anchor=(1.1,-0.45), loc='lower right', fontsize=18, ncol=2)

