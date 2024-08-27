import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import sys
import classification_algorithms as ca
from scipy.stats import spearmanr


def get_feature_label(features):
    name = 'ax' if len(features) == 1 else 'var' if np.any(
        [True if i.find('var') != -1 else False for i in features]) else 'mean'
    return name


def generate_intersubject_output(subject_type, reduced_type='mean'):
    """
    Generates and saves inter-subject model output using 'classifier'
    :param subject_type: 'control', 'patient' - reads data and features from folder
    :param reduced_type: 'ax', 'mean', 'var'
    """
    if reduced_type == 'ax':
        features = ['ax_mean']
    elif reduced_type == 'mean':
        features = ['ax_mean', 'ay_mean', 'az_mean']
    elif reduced_type == 'var':
        features = ['ax_mean', 'ay_mean', 'az_mean', 'ax_var', 'ay_var', 'az_var']
    else:
        raise Exception(f"Invalid classifier: {reduced_type}. Use 'ax', 'mean' or 'var' instead.")

    clf = RandomForestClassifier(class_weight='balanced')
    param_grid = {'n_estimators': [100, 200, 500, 1000, 1200, 1500]}
    func = ca.train_validate_and_test_intersubject

    ldf, rdf = ca.read_features(subject_type, 4)

    sys.stdout.write('Generating inter-subject output...\n')
    sys.stdout.write(f'Left/Affected arm ({ldf.subject.max()} subjects):\n')
    rfl = ca.call_machine_learning_function(ldf, func, classifier=clf,
                                            param_grid=param_grid, features=features).rename(columns={'pred': 'l'})[
        ['l']]
    sys.stdout.write(f'Right/Unaffected arm ({rdf.subject.max()} subjects:\n')
    rfr = ca.call_machine_learning_function(rdf, func, classifier=clf,
                                            param_grid=param_grid, features=features).rename(columns={'pred': 'r'})[
        ['r']]

    rf = pd.merge(rfl, rfr, on='time')
    rf.to_csv(subject_type + f'/classifier outputs/rf_inter_{get_feature_label(features)}.csv')


def generate_intrasubject_output(subject_type, reduced_type):
    """
    Generates and saves intra-subject model output using 'classifier'
    :param subject_type: 'control', 'patient' - reads data and features from folder
    :param reduced_type: 'ax', 'mean', 'var'
    """
    if reduced_type == 'ax':
        features = ['ax_mean']
    elif reduced_type == 'mean':
        features = ['ax_mean', 'ay_mean', 'az_mean']
    elif reduced_type == 'var':
        features = ['ax_mean', 'ay_mean', 'az_mean', 'ax_var', 'ay_var', 'az_var']
    else:
        raise Exception(f"Invalid classifier: {reduced_type}. Use 'ax', 'mean' or 'var' instead.")

    clf = RandomForestClassifier(class_weight='balanced')
    param_grid = {'n_estimators': [100, 200, 500, 1000, 1200, 1500]}
    func = ca.train_validate_and_test_intrasubject

    ldf, rdf = ca.read_features(subject_type, 4)

    sys.stdout.write('Generating intra-subject output...\n')
    rf = []
    sys.stdout.write('Iterating 10 times:\n')
    for i in tqdm(np.arange(10)):
        rfl = ca.call_machine_learning_function(ldf, func, classifier=clf,
                                                param_grid=param_grid, features=features).rename(columns={'pred': 'l'})[
            ['l']]
        rfr = ca.call_machine_learning_function(rdf, func, classifier=clf,
                                                param_grid=param_grid, features=features).rename(columns={'pred': 'r'})[
            ['r']]

        rf.append(pd.merge(rfl, rfr, on='time').assign(iteration=i))

    rf = pd.concat(rf)
    rf.to_csv(subject_type + f'/classifier outputs/rf_intra_{get_feature_label(features)}.csv')


def plot_reduced_model_results():
    files_sets = [['rf_intra.csv', 'rf_intra_ax.csv', 'rf_intra_mean.csv', 'rf_intra_var.csv'],
                  ['rf_inter.csv', 'rf_inter_ax.csv', 'rf_inter_mean.csv', 'rf_inter_var.csv']]
    labels = ['full', 'mean of ax', 'all mean', 'all mean and variance']
    ls = []
    left, right = ca.read_data('control')
    affected, unaffected = ca.read_data('patient')
    for files, tag in zip(files_sets, ['intra', 'inter']):
        datasets = [[right, left], [unaffected, affected]]

        for datas, typ, fignum in zip(datasets, ['control', 'patient'], [1, 3]):
            # generate results
            title = 'right' if fignum == 1 else 'unaffected'
            for data, hand, n in zip(datas, ['r', 'l'], [0, 1]):
                title = 'left' if (n == 1) and (title == 'right') else 'affected' if (n == 1) and (
                        title == 'unaffected') else title
                fignum = fignum + n
                subresults = []
                for name, label in zip(files, labels):
                    df = pd.read_csv(typ + '/classifier outputs/' + name, parse_dates=['time'], index_col='time')
                    temp = pd.merge(data[['gnd', 'subject']], df, on='time')
                    if name.find('intra') != -1:
                        for sub, subdata in temp.groupby('subject'):
                            for i, itdf in subdata.groupby('iteration'):
                                subresults += [
                                    ca.confmatrix(itdf[hand[0]], itdf['gnd']).assign(subject=sub, method=label,
                                                                                     iteration=i)]
                    else:
                        for sub, subdata in temp.groupby('subject'):
                            subresults += [
                                ca.confmatrix(subdata[hand[0]], subdata['gnd']).assign(subject=sub, method=label)]

                table = pd.concat(subresults)

                table['youden'] = table['sensitivity'] + table['specificity'] - 1
                table['type'] = title + '_' + tag
                ls.append(table)
    result = pd.concat(ls)

    cont_intra = pd.concat([result[result['type'] == 'right_intra'], result[result['type'] == 'left_intra']])
    cont_intra['type'] = 'control intra'

    cont_inter = pd.concat([result[result['type'] == 'right_inter'], result[result['type'] == 'left_inter']])
    cont_inter['type'] = 'control inter'

    pat_intra = pd.concat([result[result['type'] == 'unaffected_intra'], result[result['type'] == 'affected_intra']])
    pat_intra['type'] = 'patient intra'

    pat_inter = pd.concat([result[result['type'] == 'unaffected_inter'], result[result['type'] == 'affected_inter']])
    pat_inter['type'] = 'patient inter'

    result = pd.concat([cont_intra, pat_intra, cont_inter, pat_inter])
    result.to_csv('results/reduced_models.csv')
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.boxplot(data=result, x='type', y='youden', hue='method', notch=True, ax=ax)
    ax.set_xlim(0-0.5, 3+0.5)
    plt.xlabel(None)
    plt.xticks(fontsize=14)
    plt.ylabel('Youden Index', fontsize=14)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:], fontsize=14, bbox_to_anchor=(1.01, 1), ncol=1)
    
def get_features(data, freq):
  acc_vectors = np.array([data['ax'], data['ay'], data['az']], np.int32)
  data['anorm'] = [np.linalg.norm(acc_vectors[:, i]) for i in range(len(data))]
  axz = np.array([data['ax'], data['az']], np.int32)
  axz = np.array([np.linalg.norm(axz[:, i]) for i in range(len(data))])
  data['roll'] = np.arctan2(-data['ay'].values, data['az'].values)*(180/np.pi)

  data.index = pd.to_datetime(data.index)
  df = pd.DataFrame()
  data_list = ca.get_continuous_segments(data)
  data_list = [data.resample(str(round(1/freq, 2))+'S', label='right', closed='right') for data in data_list]
  blocks = [block for data in data_list for i,block in data if len(block)>0]

  df['ax_mean'] = [block['ax'].mean() for block in blocks]
  df['ax_var'] = [block['ax'].var() for block in blocks]
  df['ay_mean'] = [block['ay'].mean() for block in blocks]
  df['ay_var'] = [block['ay'].var() for block in blocks]
  df['az_mean'] = [block['az'].mean() for block in blocks]
  df['az_var'] = [block['az'].var() for block in blocks]
  
  df['a2_mean'] = [block['anorm'].mean() for block in blocks]
  df['a2_var'] = [block['anorm'].var() for block in blocks]
  df['a2_min'] = [block['anorm'].min() for block in blocks]
  df['a2_max'] = [block['anorm'].max() for block in blocks]
  df['entropy'] = [ca.entropy(block['anorm'].values) for block in blocks]
  df['pitch'] = [block['pitch'][-1] for block in blocks]
  df['roll'] = [block['roll'][-1] for block in blocks]
  df['yaw'] = [block['yaw'][-1] for block in blocks]
  df['delta_yaw'] = [block['yaw'].max()-block['yaw'].min() for block in blocks]
  
  df['subject'] = [block['subject'][0] for block in blocks]
  df['t_inst'] = [block['gnd'][-1] for block in blocks]
  df['t_maj'] = [np.round(block['gnd'].mean()) for block in blocks]
  df['t_mid'] = [block['gnd'][int(len(block)/2)] for block in blocks]
  df['time'] = [block.index[-1] for block in blocks]
  
  df = df.set_index('time')
  df.index = pd.to_datetime(df.index)
  
  cdf = ca.compute_vector_magnitude(data)
  df = pd.merge_asof(cdf.fillna(0), df, on='time')

  df = df.dropna(subset=['ax_var', 'ay_var', 'az_var'])

  return df.set_index('time')
  
  
def compute_corr_matrix():
    left, right = ca.read_data('control')
    affected, unaffected = ca.read_data('patient')
    df = pd.concat([get_features(right, 1), get_features(left, 1), get_features(unaffected, 1), get_features(affected, 1)])
    
    rf_features = ['ax_mean', 'ax_var', 'ay_mean', 'ay_var', 'az_mean', 'az_var','a2_mean', 'a2_var', 'a2_max', 'a2_min',  'entropy']
    features = ['pitch', 'delta_yaw', 'counts']#, 'roll']
    
    temp = df
    corr = np.vstack(([r for r,p in [spearmanr(temp['pitch'], temp[feat]) for feat in rf_features]], 
                      [r for r,p in [spearmanr(temp['delta_yaw'], temp[feat]) for feat in rf_features]],
                      [r for r,p in [spearmanr(temp['counts'], temp[feat]) for feat in rf_features]],
                    #   [r for r,p in [spearmanr(temp['roll'], temp[feat]) for feat in rf_features]]
                      ))
    corr = pd.DataFrame(corr, columns=rf_features)
    corr['feature'] = features
    xnames = [r'$\overline{a_x}$', r'$\sigma^2(a_x)$', r'$\overline{a_y}$', r'$\sigma^2(a_y)$', r'$\overline{a_z}$', r'$\sigma^2(a_z)$', r'$\overline{||a||_2}$', r'$\sigma^2(||a||_2)$', r'$max(||a||_2)$', r'$min(||a||_2)$', r'$entropy(||a||_2)$']
    ynames = [r'$pitch$', r'$\Delta yaw$', r'$counts$']#, r'$roll$']
    
    plt.figure(figsize=(20, 2))
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    g = sns.heatmap(data=(corr.set_index('feature')), cmap=cmap, annot=True, fmt='.2f', annot_kws={'fontsize':15, 'color':'k'}, 
                xticklabels=xnames, yticklabels=ynames, cbar_kws={'pad':0.005, 'aspect':5, 'ticks':[-0.4,0,0.5,0.9]})
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=14)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=14)
    plt.yticks(rotation=0)
    plt.ylabel(None)
    return corr
    
def compute_gyro_features(data, freq):
  data.index = pd.to_datetime(data.index)
  df = pd.DataFrame()
  data_list = ca.get_continuous_segments(data)
  data_list = [data.resample(str(round(1/freq, 2))+'S', label='right', closed='right') for data in data_list]
  blocks = [block for data in data_list for i,block in data if len(block)>0]

  df['ax_mean'] = [block['ax'].mean() for block in blocks]
  df['ax_var'] = [block['ax'].var() for block in blocks]
  df['ay_mean'] = [block['ay'].mean() for block in blocks]
  df['ay_var'] = [block['ay'].var() for block in blocks]
  df['az_mean'] = [block['az'].mean() for block in blocks]
  df['az_var'] = [block['az'].var() for block in blocks]
  
  df['gx_mean'] = [block['gx'].mean() for block in blocks]
  df['gx_var'] = [block['gx'].var() for block in blocks]
  df['gy_mean'] = [block['gy'].mean() for block in blocks]
  df['gy_var'] = [block['gy'].var() for block in blocks]
  df['gz_mean'] = [block['gz'].mean() for block in blocks]
  df['gz_var'] = [block['gz'].var() for block in blocks]
  
  df['subject'] = [block['subject'][0] for block in blocks]
  df['t_inst'] = [block['gnd'][-1] for block in blocks]
  df['t_maj'] = [np.round(block['gnd'].mean()) for block in blocks]
  df['t_mid'] = [block['gnd'][int(len(block)/2)] for block in blocks]
  df['time'] = [block.index[-1] for block in blocks]
  
  df = df.set_index('time')
  df.index = pd.to_datetime(df.index)

  df = df.dropna(subset=['ax_var', 'ay_var', 'az_var'])

  return df
  
def generate_results_with_gyro():
    lraw, rraw = ca.read_data('control')
    araw, uraw = ca.read_data('patient')
    
    left = compute_gyro_features(lraw, 4)
    right = compute_gyro_features(rraw, 4)
    affected = compute_gyro_features(araw, 4)
    unaffected = compute_gyro_features(uraw, 4)
    accl = ['ax_mean', 'ay_mean', 'az_mean', 'ax_var','ay_var', 'az_var']
    with_gyro = accl + ['gx_mean', 'gy_mean', 'gz_mean', 'gx_var','gy_var', 'gz_var']
    
    linter = ca.train_and_test_intersubject(left, features=accl, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    rinter = ca.train_and_test_intersubject(right, features=accl, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    uinter = ca.train_and_test_intersubject(affected, features=accl, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    ainter = ca.train_and_test_intersubject(unaffected, features=accl, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    
    lginter = ca.train_and_test_intersubject(left, features=with_gyro, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    rginter = ca.train_and_test_intersubject(right, features=with_gyro, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    uginter = ca.train_and_test_intersubject(affected, features=with_gyro, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    aginter = ca.train_and_test_intersubject(unaffected, features=with_gyro, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    
    lintra = ca.train_and_test_intrasubject(left, features=accl, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    rintra = ca.train_and_test_intrasubject(right, features=accl, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    uintra = ca.train_and_test_intrasubject(affected, features=accl, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    aintra = ca.train_and_test_intrasubject(unaffected, features=accl, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    
    lgintra = ca.train_and_test_intrasubject(left, features=with_gyro, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    rgintra = ca.train_and_test_intrasubject(right, features=with_gyro, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    ugintra = ca.train_and_test_intrasubject(affected, features=with_gyro, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    agintra = ca.train_and_test_intrasubject(unaffected, features=with_gyro, target='t_mid', classifier=RandomForestClassifier(class_weight='balanced'))
    
    methods_dict = {'rinter':rinter, 'linter':linter, 'uinter':uinter, 'ainter':ainter,
                    'rginter':rginter, 'lginter':lginter, 'uginter':uginter, 'aginter':aginter, 
                    'rintra':rintra, 'lintra':lintra, 'uintra':uintra, 'aintra':aintra, 
                    'rgintra':rgintra, 'lgintra':lgintra, 'uginter':ugintra, 'agintra':agintra}
    subresults = []
    for name, df in methods_dict.items():
        for sub, subdata in df.groupby('subject'):
            subresults += [ca.confmatrix(subdata['pred'], subdata['t_mid']).assign(subject=sub, method=name)]

    table = pd.concat(subresults)
    table.to_csv('results/with_gyro.csv')
