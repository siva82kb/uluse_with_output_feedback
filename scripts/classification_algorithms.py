import pandas as pd
from ahrs.filters import Mahony, Madgwick
from ahrs.common import orientation
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, PredefinedSplit, GridSearchCV
from sklearn.neighbors._kde import KernelDensity
from scipy import stats
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import sys


def get_continuous_segments(df):
    # returns a list of continuous sections (as dataframes) from the original dataframe

    time_diff = np.array([pd.Timedelta(diff).total_seconds()
                          for diff in np.diff(df.index.values)])
    inx = np.sort(np.append(np.where(time_diff > 60)[0], -1))
    dfs = [df.iloc[inx[i] + 1:inx[i + 1] + 1]
           if i + 1 < len(inx)
           else df.iloc[inx[-1] + 1:]
           for i in np.arange(len(inx))]
    return dfs


def confmatrix(pred, target):
    n = len(pred)
    notpred = np.logical_not(pred)
    nottarget = np.logical_not(target)
    tp = (np.logical_and(pred, target).sum()) / n
    fp = (np.logical_and(pred, nottarget).sum()) / n
    fn = (np.logical_and(notpred, target).sum()) / n
    tn = (np.logical_and(notpred, nottarget).sum()) / n
    acc = (tp + tn)
    pfa = tp + fn
    pother = tp + fp
    x = ((pfa + pother) / 2)
    pe = 2 * x * (1 - x)
    gwet = (acc - pe) / (1 - pe)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (2 * sens * prec) / (sens + prec) if (sens + prec) else np.nan
    bal = (sens + spec) / 2
    results = {'true positive': tp,
               'false positive': fp,
               'false negative': fn,
               'true negative': tn,
               'accuracy': acc,
               'gwets ac1 score': gwet,
               'sensitivity': sens,
               'specificity': spec,
               'precision': prec,
               'f1 score': f1,
               'balanced accuracy': bal}
    results = pd.DataFrame.from_dict({k: [v] for k, v in results.items()})
    return results


def bandpass(x, fs=50, order=4):
    sos = signal.butter(order, [0.25, 2.5], 'bandpass', fs=fs, output='sos', analog=False)
    filtered = signal.sosfilt(sos, x)
    return filtered


def resample(df, current_fs, new_fs):
    dfs = get_continuous_segments(df)
    dfs = [df.resample(str(round(1 / new_fs, 2)) + 'S', label='right', closed='right').mean()
           for df in dfs]
    df = pd.concat(dfs)
    df.index.name = 'time'
    return df


def threshold_filter(df):
    dfs = get_continuous_segments(df)
    dfs = [df.resample('0.1S').last() for df in dfs]
    df = pd.concat(dfs)
    df = df.fillna(0)

    op_df = pd.DataFrame(index=df.index)

    acc = np.array(df[['ax', 'az']].apply(lambda x: (x / (0.01664)).astype(int)))

    op_df['cx'] = acc[:, 0]
    op_df['cz'] = acc[:, 1]
    op_dfs = get_continuous_segments(op_df)
    op_dfs = [op_df.resample('2S').sum() for op_df in op_dfs]
    op_df = pd.concat(op_dfs)

    c = np.array(op_df[['cx', 'cz']])
    counts = np.linalg.norm(c, axis=1)
    scores = [1 if c > 2 else 0 for c in counts]

    op_df['counts'] = counts
    op_df['ac'] = scores
    return op_df[['ac']]


def compute_activity_counts(df):
    op_df = pd.DataFrame(index=df.index)

    g = np.array([0, 0, 1])
    ae = np.empty([len(df), 3])

    mg = Mahony(frequency=50, beta=0.5)
    q = np.tile([1., 0., 0., 0.], (len(df), 1))
    gyr = np.array(df[['gx', 'gy', 'gz']])
    acc = np.array(df[['ax', 'ay', 'az']])
    mag = np.array(df[['mx', 'my', 'mz']])

    r = orientation.q2R(mg.updateMARG(q[0], gyr[0], acc[0], mag[0]))
    ae[0] = np.matmul(r, acc[0]) - g

    for i in range(1, len(df)):
        q[i] = mg.updateMARG(q[i - 1], gyr[i], acc[i], mag[i])
        r = orientation.q2R(q[i])
        ae[i] = np.matmul(r, acc[i]) - g

    # filter acceleration magnitudes
    a_mag = [np.linalg.norm(x) for x in ae]
    op_df['a_mag'] = np.absolute(bandpass(a_mag))

    op_df = resample(op_df, 50, 1)
    op_df = op_df.fillna(0)

    op_df['counts'] = (op_df['a_mag'].apply(lambda x: x / 0.01664).astype(int))
    return op_df


def compute_vector_magnitude(df: pd.DataFrame, fs: float=30) -> pd.DataFrame:
    """Computes the vector magnitude from the IMU data.
    """
    df = df.loc[:, ['ax', 'ay', 'az', 'gx', 'gy', 'gz']]
    df = resample(df, 50, 30)
    op_df = pd.DataFrame(index=df.index)

    gyr = np.array(df[['gx', 'gy', 'gz']])
    acc = np.array(df[['ax', 'ay', 'az']])

    g = np.array([0, 0, 1])
    ae = np.empty([len(acc), 3])

    mg = Madgwick(frequency=fs, beta=0.5)
    q = np.tile([1., 0., 0., 0.], (len(acc), 1))

    r = orientation.q2R(mg.updateIMU(q[0], gyr[0], acc[0]))
    ae[0] = np.matmul(r, acc[0]) - g

    for i in range(1, len(acc)):
        q[i] = mg.updateIMU(q[i - 1], gyr[i], acc[i])
        r = orientation.q2R(q[i])
        ae[i] = np.matmul(r, acc[i]) - g

    op_df['ax'] = bandpass(np.nan_to_num(ae[:, 0]), fs=fs)
    op_df['ay'] = bandpass(np.nan_to_num(ae[:, 1]), fs=fs)
    op_df['az'] = bandpass(np.nan_to_num(ae[:, 2]), fs=fs)
    op_df = resample(op_df, fs, 10)

    op_df['ax'] = np.where(np.absolute(op_df['ax'].values) < 0.068, 0, op_df['ax'].values) / 0.01664
    op_df['ay'] = np.where(np.absolute(op_df['ay'].values) < 0.068, 0, op_df['ay'].values) / 0.01664
    op_df['az'] = np.where(np.absolute(op_df['az'].values) < 0.068, 0, op_df['az'].values) / 0.01664

    dfs = get_continuous_segments(op_df)
    dfs = [df.resample(str(1) + 'S').sum() for df in dfs]
    op_df = pd.concat(dfs)
    op_df.index.name = 'time'
    op_df = op_df.fillna(0)

    op_df['a_mag'] = [np.linalg.norm(x) for x in np.array(op_df[['ax', 'ay', 'az']])]
    op_df['counts'] = [np.round(x) for x in op_df['a_mag'].rolling(5).mean()]
    return op_df[['counts']]


def calculate_gm(pitch, yaw, fun_range=30, min_mvt=30):
    return 1.0 if (np.all(np.abs(pitch) < fun_range) and (
            max(yaw) - min(yaw) + max(pitch) - min(pitch)) > min_mvt) else 0.0


def get_gm_scores(data):
    op_dfs = []
    for df in get_continuous_segments(data[['yaw', 'pitch']]):
        df['time'] = df.index
        df.set_index(np.arange(len(df)), inplace=True)
        op = np.array([(df.loc[i + 100, 'time'],
                        calculate_gm(df.loc[i:i + 100, 'pitch'], df.loc[i:i + 100, 'yaw']))
                       for i in np.arange(0, len(df) - 100, 25)])
        op_df = pd.DataFrame(op, columns=['time', 'gm'])
        op_dfs.append(op_df)

    gm = pd.concat(op_dfs)
    return gm


def get_gm_modified(data):
    pitch = resample(data[['pitch']], 50, 1)
    counts = compute_vector_magnitude(data)
    gmac = pd.merge(pitch, counts, on='time')
    gmac['pred'] = [1 if np.abs(pitch) < 30 and count > 0 else 0 for pitch, count in zip(gmac['pitch'], gmac['counts'])]
    return gmac.reset_index()


def entropy(data):
    data = np.array(data).reshape(-1, 1)
    # estimate pdf using KDE with gaussian kernel
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
    log_p = kde.score_samples(data)
    return stats.entropy(np.exp(log_p))


def compute_features(data, freq):
    acc_vectors = np.array([data['ax'], data['ay'], data['az']], np.int32)
    data['anorm'] = [np.linalg.norm(acc_vectors[:, i]) for i in range(len(data))]

    data.index = pd.to_datetime(data.index)
    df = pd.DataFrame()
    data_list = get_continuous_segments(data)
    data_list = [data.resample(str(round(1 / freq, 2)) + 'S', label='right', closed='right') for data in data_list]
    blocks = [block for data in data_list for i, block in data if len(block) > 0]

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
    df['entropy'] = [entropy(block['anorm'].values) for block in blocks]

    df['subject'] = [block['subject'][0] for block in blocks]
    df['task'] = [block['task'][0] for block in blocks]
    df['t_inst'] = [block['gnd'][-1] for block in blocks]
    df['t_maj'] = [np.round(block['gnd'].mean()) for block in blocks]
    df['t_mid'] = [block['gnd'][int(len(block) / 2)] for block in blocks]
    df['time'] = [block.index[-1] for block in blocks]

    df = df.set_index('time')
    df.index = pd.to_datetime(df.index)

    df = df.dropna(subset=['ax_var', 'ay_var', 'az_var'])
    return df


def split_intrasubject(subdata, features, target, n_splits):
    folds = []
    remain_inx = np.array(range(len(subdata))).astype(int)
    for fold in range(2, n_splits)[::-1]:
        remain_inx, test_inx, _, _ = train_test_split(remain_inx,
                                                      subdata[target].values[remain_inx].reshape(-1),
                                                      test_size=1 / fold,
                                                      stratify=subdata[target].values[remain_inx].reshape(-1))
        fold_split = (np.setdiff1d(range(len(subdata)), np.array(test_inx)), test_inx)
        folds.append(fold_split)
    folds.append((np.setdiff1d(range(len(subdata)), np.array(remain_inx)), remain_inx))
    return folds


def split_intersubject(data):
    df = data.copy()
    df['inx'] = range(len(df))
    folds = []
    for sub, subdata in df.groupby('subject'):
        val_inx = subdata['inx'].values
        train_inx = df[df['subject'] != sub]['inx'].values
        fold_split = (train_inx, val_inx)
        folds.append(fold_split)
    return folds


def train_and_test_intersubject(df, features, target, classifier, param_grid=None):
    new_subs = []

    for sub, subdata in df.groupby('subject'):
        traindata = df[df['subject'] != sub]
        x_train = traindata[features].values
        x_test = subdata[features].values
        y_train = traindata[target].values.reshape(-1)
        y_test = subdata[target].values.reshape(-1)
        classifier.fit(x_train, y_train)

        subdata['pred'] = classifier.predict(x_test)
        new_subs.append(subdata)
    return pd.concat(new_subs)


def train_and_test_intrasubject(df, features, target, classifier, param_grid=None):
    col = features + ['time', 'subject']
    new_subs = []

    for sub, subdata in df.groupby('subject'):
        traindata = pd.DataFrame()
        testdata = pd.DataFrame()
        subdata = subdata.reset_index()

        x_train, x_test, y_train, y_test = train_test_split(subdata[col].values,
                                                            subdata[target].values.reshape(-1),
                                                            test_size=0.2,
                                                            stratify=subdata[target].values.reshape(-1))

        traindata[col] = x_train
        testdata[col] = x_test
        traindata[target] = y_train.reshape((-1,1))
        testdata[target] = y_test.reshape(-1,1)
        classifier.fit(x_train[:, :-2], y_train)

        testdata['pred'] = classifier.predict(x_test[:, :-2])
        new_subs.append(testdata)

    return pd.concat(new_subs).set_index('time')


def train_validate_and_test_intersubject(df, features, target, classifier, param_grid):
    new_subs = []
    sublist = df.groupby('subject')

    for sub, subdata in tqdm(sublist):
        # the subject sub is taken as the test set
        traindata = df[df['subject'] != sub]

        x_train = traindata[features].values
        x_test = subdata[features].values

        y_train = traindata[target].values.reshape(-1)
        y_test = subdata[target].values.reshape(-1)

        # tune hyperparameters by cross validation
        split = split_intersubject(traindata)
        grid_search = GridSearchCV(classifier, param_grid, cv=split)

        grid_search.fit(x_train, y_train)

        best_classifier = grid_search.best_estimator_

        # run the classifier on the test set
        subdata['pred'] = best_classifier.predict(x_test)
        new_subs.append(subdata)

    return pd.concat(new_subs)


def train_validate_and_test_intrasubject(df, features, target, classifier, param_grid):
    col = features + ['time', 'subject', 'task']
    new_subs = []
    sublist = df.groupby('subject')

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
        traindata[target] = y_train.reshape((-1,1))
        testdata[target] = y_test.reshape((-1,1))

        # tune hyperparameters by cross validation
        split = split_intrasubject(traindata, features, target, 4)
        grid_search = GridSearchCV(classifier, param_grid, cv=split)

        grid_search.fit(traindata[features].values, traindata[target].values.reshape(-1))

        best_classifier = grid_search.best_estimator_

        # run the classifier on the test set
        testdata['pred'] = best_classifier.predict(testdata[features].values)
        new_subs.append(testdata)

    return pd.concat(new_subs).set_index('time')


def read_data(subject_type):
    """
    Reads raw data from 'subject_type' folder
    """
    if subject_type == 'patient':
        left = pd.read_csv(subject_type + '/data/affected.csv', parse_dates=['time'], index_col='time')
        right = pd.read_csv(subject_type + '/data/unaffected.csv', parse_dates=['time'], index_col='time')
    elif subject_type == 'control':
        left = pd.read_csv(subject_type + '/data/left.csv', parse_dates=['time'], index_col='time')
        right = pd.read_csv(subject_type + '/data/right.csv', parse_dates=['time'], index_col='time')
    else:
        raise Exception(f"Invalid parameter: {subject_type}. Use 'control' or 'patient' instead.")
    return left, right


def read_features(subject_type, freq=4):
    """
    Reads features from 'subject_type' folder
    """
    if not os.path.exists(subject_type):
        raise Exception(f'Invalid parameter: {subject_type}')
    if not os.path.exists(subject_type + f'/features/left_{freq}hz.csv') or \
            not os.path.exists(subject_type + f'/features/right_{freq}hz.csv'):
        sys.stdout.write('Feature files not found.')
        generate_and_save_features(subject_type, freq)

    left = pd.read_csv(subject_type + f'/features/left_{freq}hz.csv', parse_dates=['time'], index_col='time')
    right = pd.read_csv(subject_type + f'/features/right_{freq}hz.csv', parse_dates=['time'], index_col='time')

    return left, right


def generate_and_save_features(subject_type, freq=4):
    # features for machine learning
    sys.stdout.write('Generating features (4hz) for machine learning...')
    left, right = read_data(subject_type)
    ldf = compute_features(left, freq)
    ldf.to_csv(subject_type + f'/features/left_{freq}hz.csv')
    rdf = compute_features(right, freq)
    rdf.to_csv(subject_type + f'/features/right_{freq}hz.csv')
    sys.stdout.write('Done \n')


def generate_threshold_filter_output(subject_type):
    """
    Generates and saves classifier output using threshold filter for 'subject_type' dataset
    :param subject_type: 'control', 'patient' - reads data and features from folder
    """
    left, right = read_data(subject_type)

    # threshold filter
    sys.stdout.write('Generating threshold filter output...')
    tfl = threshold_filter(left).rename(columns={'ac': 'l'})
    tfr = threshold_filter(right).rename(columns={'ac': 'r'})

    tf = pd.merge(tfl, tfr, on='time')
    tf.to_csv(subject_type + '/classifier outputs/tf.csv')
    sys.stdout.write('Done \n')


def generate_activity_counts_output(subject_type):
    """
    Generates and saves classifier output using activity counts for 'subject_type' dataset
    :param subject_type: 'control', 'patient' - reads data and features from folder
    """
    left, right = read_data(subject_type)
    # activity counts
    sys.stdout.write('Generating activity counts...')
    counts_left = compute_activity_counts(left).rename(columns={'counts': 'counts_left'})['counts_left']
    counts_right = compute_activity_counts(right).rename(columns={'counts': 'counts_right'})['counts_right']

    ac = pd.merge(counts_left, counts_right, on='time')

    num = ac['counts_right'].values - ac['counts_left'].values
    den = ac['counts_right'].values + ac['counts_left'].values
    empty = np.empty(len(num))
    empty[:] = np.nan
    ac['laterality'] = np.divide(num, den, out=empty, where=den != 0)

    ac['r'] = np.zeros(len(ac))
    ac['l'] = np.zeros(len(ac))

    ac.loc[ac['laterality'] > -0.95, 'r'] = 1
    ac.loc[ac['laterality'] < 0.95, 'l'] = 1
    ac.to_csv(subject_type + '/classifier outputs/ac.csv')
    sys.stdout.write('Done \n')


def generate_vector_magnitude_output(subject_type):
    """
    Generates and saves classifier output using vector magnitude for 'subject_type' dataset
    :param subject_type: 'control', 'patient' - reads data and features from folder
    """
    left, right = read_data(subject_type)
    # vector magnitude
    sys.stdout.write('Generating vector magnitude output...')
    counts_left = compute_vector_magnitude(left).rename(columns={'counts': 'counts_left'})['counts_left']
    counts_right = compute_vector_magnitude(right).rename(columns={'counts': 'counts_right'})['counts_right']

    vm = pd.merge(counts_left, counts_right, on='time')
    vm = vm.fillna(0)
    vm['r'] = np.zeros(len(vm))
    vm['l'] = np.zeros(len(vm))

    vm.loc[vm['counts_left'] > 0, 'l'] = 1
    vm.loc[vm['counts_right'] > 0, 'r'] = 1
    vm.to_csv(subject_type + '/classifier outputs/vm.csv')
    sys.stdout.write('Done \n')


def generate_gm_score_output(subject_type):
    """
    Generates and saves classifier output using GM score for 'subject_type' dataset
    :param subject_type: 'control', 'patient' - reads data and features from folder
    """
    left, right = read_data(subject_type)
    # GM score
    sys.stdout.write('Generating GM scores...')
    gml = get_gm_scores(left).rename(columns={'gm': 'l'})
    gmr = get_gm_scores(right).rename(columns={'gm': 'r'})

    gm = pd.merge_asof(gml, gmr, on='time', tolerance=pd.Timedelta('500ms'), direction='nearest')
    gm.to_csv(subject_type + '/classifier outputs/gm.csv')
    sys.stdout.write('Done \n')


def generate_hybrid_gmac_output(subject_type):
    """
    Generates and saves classifier output using modified GM score for 'subject_type' dataset
    :param subject_type: 'control', 'patient' - reads data and features from folder
    """
    left, right = read_data(subject_type)
    # modified GM score
    sys.stdout.write('Generating modified GM scores...')
    gml = get_gm_modified(left).rename(columns={'pred': 'l'})
    gmr = get_gm_modified(right).rename(columns={'pred': 'r'})

    gm = pd.merge(gml, gmr, on='time').set_index('time')
    gm.to_csv(subject_type + '/classifier outputs/gmac.csv')
    sys.stdout.write('Done \n')


def call_machine_learning_function(data, function, features='all', **kwargs):
    if features == 'all':
        features = ['ax_mean', 'ax_var', 'ay_mean', 'ay_var', 'az_mean', 'az_var',
                    'a2_mean', 'a2_var', 'a2_min', 'a2_max', 'entropy']
    target = ['t_mid']
    return function(data, features, target, **kwargs)


def generate_intersubject_output(subject_type, classifier):
    """
    Generates and saves inter-subject model output using 'classifier'
    :param subject_type: 'control', 'patient' - reads data and features from folder
    :param classifier: 'rf', 'svm', 'mlp'
    """
    if classifier == 'rf':
        clf = RandomForestClassifier(class_weight='balanced')
        param_grid = {'n_estimators': [100, 200, 500, 1000, 1200, 1500]}
        func = train_validate_and_test_intersubject
    elif classifier == 'svm':
        clf = svm.SVC(class_weight='balanced')
        param_grid = {'C': [1, 10, 100, 1000],
                      'gamma': ['auto', 'scale']}
        func = train_validate_and_test_intersubject
    elif classifier == 'mlp':
        num_features = 11
        l1 = int((2 / 3) * num_features + 1)
        l2 = int((2 / 3) * l1 + 1)
        l3 = int((2 / 3) * l2 + 1)
        clf = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), max_iter=2000)
        param_grid = None
        func = train_and_test_intersubject
    else:
        raise Exception(f"Invalid classifier: {classifier}. Use 'rf', 'svm' or 'mlp' instead.")

    ldf, rdf = read_features(subject_type, 4)

    sys.stdout.write('Generating inter-subject output...\n')
    sys.stdout.write(f'Left/Affected arm ({ldf.subject.max()} subjects):\n')
    rfl = call_machine_learning_function(ldf, func, classifier=clf,
                                         param_grid=param_grid).rename(columns={'pred': 'l'})[['l']]
    sys.stdout.write(f'Right/Unaffected arm ({rdf.subject.max()} subjects:\n')
    rfr = call_machine_learning_function(rdf, func, classifier=clf,
                                         param_grid=param_grid).rename(columns={'pred': 'r'})[['r']]

    rf = pd.merge(rfl, rfr, on='time')
    rf.to_csv(subject_type + f'/classifier outputs/{classifier}_inter.csv')


def generate_intrasubject_output(subject_type, classifier):
    """
    Generates and saves intra-subject model output using 'classifier'
    :param subject_type: 'control', 'patient' - reads data and features from folder
    :param classifier: 'rf', 'svm', 'mlp'
    """
    if classifier == 'rf':
        clf = RandomForestClassifier(class_weight='balanced')
        param_grid = {'n_estimators': [100, 200, 500, 1000, 1200, 1500]}
        func = train_validate_and_test_intrasubject
    elif classifier == 'svm':
        clf = svm.SVC(class_weight='balanced')
        param_grid = {'C': [1, 10, 100, 1000],
                      'gamma': ['auto', 'scale']}
        func = train_validate_and_test_intrasubject
    elif classifier == 'mlp':
        num_features = 11
        l1 = int((2 / 3) * num_features + 1)
        l2 = int((2 / 3) * l1 + 1)
        l3 = int((2 / 3) * l2 + 1)
        clf = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), max_iter=2000)
        param_grid = None
        func = train_and_test_intrasubject
    else:
        raise Exception(f"Invalid classifier: {classifier}. Use 'rf', 'svm' or 'mlp' instead.")

    ldf, rdf = read_features(subject_type, 4)

    sys.stdout.write('Generating intra-subject output...\n')
    rf = []
    sys.stdout.write('Iterating 10 times:\n')
    for i in tqdm(np.arange(10)):
        rfl = call_machine_learning_function(ldf, func, classifier=clf,
                                             param_grid=param_grid).rename(columns={'pred': 'l'})[['l']]
        rfr = call_machine_learning_function(rdf, func, classifier=clf,
                                             param_grid=param_grid).rename(columns={'pred': 'r'})[['r']]

        rf.append(pd.merge(rfl, rfr, on='time').assign(iteration=i))

    rf = pd.concat(rf)
    rf.to_csv(subject_type + f'/classifier outputs/{classifier}_intra.csv')


def plot_roc(methods_dict, data, hand, title, legend=True):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title, fontsize=15)

    subresults = []
    for name, df in methods_dict.items():
        temp = pd.merge(data[['gnd', 'subject']], df, on='time')
        if name.find('intra') != -1:
            for sub, subdata in temp.groupby('subject'):
                for i, itdf in subdata.groupby('iteration'):
                    subresults += [confmatrix(itdf[hand[0]], itdf['gnd']).assign(subject=sub, method=name, iteration=i)]
        else:
            for sub, subdata in temp.groupby('subject'):
                subresults += [confmatrix(subdata[hand[0]], subdata['gnd']).assign(subject=sub, method=name)]

    table = pd.concat(subresults)

    for met, metres in table.groupby('method'):
        plt.errorbar(1 - metres['specificity'].mean(),
                     metres['sensitivity'].mean(),
                     yerr=metres['sensitivity'].std(),
                     xerr=metres['specificity'].std(), alpha=0.5)
        plt.scatter(1 - metres['specificity'].mean(),
                    metres['sensitivity'].mean(),
                    label=met)

    plt.plot(np.arange(-0.5, 1.5, 1 / 100), np.arange(-0.5, 1.5, 1 / 100), 'k--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('1-Specificity', fontsize=15)
    plt.ylabel('Sensitivity', fontsize=15)
    plt.grid()
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    return table


def get_outputs(subject_type, names=[]):
    dfs = [pd.read_csv(subject_type + '/classifier outputs/' + name, parse_dates=['time'], index_col='time') for name in names]
    names = [name[:-4] for name in names]
    return dict(zip(names, dfs))


def plot_comparison_roc():
    plt.figure(figsize=(15, 3))
    left, right = read_data('control')
    affected, unaffected = read_data('patient')
    datasets = [[right, left], [unaffected, affected]]
    files = ['ac.csv', 'vm.csv', 'gm.csv', 'gmac.csv', 'rf_intra.csv', 'rf_inter.csv', 'svm_intra.csv',
             'svm_inter.csv', 'mlp_intra.csv', 'mlp_inter.csv']
    labels = ['Activity Counts', 'Vector Magnitude', 'GM Score', 'GMAC', 'RF intra', 'RF inter', 'SVM intra',
              'SVM inter', 'MLP intra', 'MLP inter']
    markers = ['o', 'X', '^', 's', '*', '*', 'P', 'P', 'D', 'D']
    colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:green', 'tab:orange', 'tab:green', 'tab:orange',
              'tab:green', 'tab:orange']

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
                                confmatrix(itdf[hand[0]], itdf['gnd']).assign(subject=sub, method=label, iteration=i)]
                else:
                    for sub, subdata in temp.groupby('subject'):
                        subresults += [confmatrix(subdata[hand[0]], subdata['gnd']).assign(subject=sub, method=label)]

            table = pd.concat(subresults)
            table['type'] = title
            plt.subplot(1, 4, fignum)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title(title, fontsize=14)
            for label, m, c in zip(labels, markers, colors):
                metres = table[table['method'] == label]
                plt.errorbar(1 - metres['specificity'].mean(),
                             metres['sensitivity'].mean(),
                             yerr=metres['sensitivity'].std(),
                             xerr=metres['specificity'].std(), alpha=0.3, linewidth=2, color=c)
                plt.scatter(1 - metres['specificity'].mean(),
                            metres['sensitivity'].mean(),
                            s=40, label=label, marker=m, color=c)
            plt.plot(np.arange(-0.5, 1.5, 1 / 100), np.arange(-0.5, 1.5, 1 / 100), 'k--', alpha=0.7)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.xlabel('1-Specificity', fontsize=12)
            plt.ylabel('Sensitivity', fontsize=12)
            plt.grid()
    plt.subplots_adjust(wspace=0.3)
    plt.legend(bbox_to_anchor=(-4, -0.2), loc='upper left', fontsize=12, ncol=5)
    


def plot_youden_boxplot():
    left, right = read_data('control')
    affected, unaffected = read_data('patient')
    datasets = [[right, left], [unaffected, affected]]
    files = ['ac.csv', 'vm.csv', 'gm.csv', 'gmac.csv', 'rf_intra.csv',
             'rf_inter.csv', 'svm_intra.csv', 'svm_inter.csv', 'mlp_intra.csv', 'mlp_inter.csv']
    labels = ['Activity Counts', 'Vector Magnitude', 'GM Score', 'GMAC', 'RF intra',
              'RF inter', 'SVM intra', 'SVM inter', 'MLP intra', 'MLP inter']
    ls = []
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
                                confmatrix(itdf[hand[0]], itdf['gnd']).assign(subject=sub, method=label, iteration=i)]
                else:
                    for sub, subdata in temp.groupby('subject'):
                        subresults += [confmatrix(subdata[hand[0]], subdata['gnd']).assign(subject=sub, method=label)]

            table = pd.concat(subresults)
            ls.append(table)
    df = pd.concat(ls)
    df['youden'] = df['sensitivity'] + df['specificity'] - 1
    df.to_csv('results/comparison.csv')
    plt.figure(figsize=(10, 2.8))
    order = df.groupby('method').mean().sort_values('youden').index
    colors = {'Activity Counts': 'tab:blue', 'Vector Magnitude': 'tab:orange', 'GM Score': 'tab:green',
              'GMAC': 'tab:pink', 'RF intra': 'tab:red', 'RF inter': 'firebrick', 'SVM intra': 'tab:purple',
              'SVM inter': 'mediumpurple', 'MLP intra': 'tab:brown', 'MLP inter': 'sienna'}
    xlabels_new = [label.replace(' ', ' \n') for label in order]
    plt.axvspan(-0.5, 3.5, facecolor='tab:blue', alpha=0.1, zorder=-1)
    plt.axvspan(3.5, 6.5, facecolor='tab:orange', alpha=0.1, zorder=-1)
    plt.axvspan(6.5, 9.5, facecolor='tab:green', alpha=0.1, zorder=-1)
    ax = sns.boxplot(data=df, x='method', y='youden', order=order, width=0.7, color='white', notch=True, 
                      medianprops=dict(color='k'), flierprops=dict(markeredgecolor='k'), capprops=dict(color='k'),
                      whiskerprops=dict(color='k'), boxprops=dict(edgecolor='k'))
    plt.text(1, 1.1, 'traditional', fontsize=12)
    plt.text(4.8, 1.1, 'inter', fontsize=12)
    plt.text(7.8, 1.1, 'intra', fontsize=12)
    ax.set_xlabel(None)
    ax.set_ylabel('Youden Index', fontsize=12)
    ax.set_xticklabels(xlabels_new, fontsize=12)
    ax.yaxis.set_tick_params(labelsize=8)
