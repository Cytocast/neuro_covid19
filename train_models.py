import json
from itertools import product
import dtreeviz
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
import shap

# current dir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(f'{ROOT_DIR}/covid_proteomics.csv')
# shuffle rows
data = data.sample(frac=1, random_state=11)

SFS_MODELS = [LogisticRegression]
CLS_MODELS = [LogisticRegression, DecisionTreeClassifier]
FEATURE_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DATA_SUBSETS = [None, 'medulla', 'csf', 'cortex']

# Uncomment to run with tissue specific proteins excluded
# DATA_SUBSET_FEATURE_FILTERS = {
#     'medulla': [
#         'Q9HAN9',
#         'Q16775',
#         'P53634',
#         'P28325',
#         'P01135',
#         'P80511',
#         'P78423',
#         'P50225'],
#     'csf': [
#         'Q16719',
#         'P80511'],
#     'cortex': [
#         'Q6NW40',
#         'Q16775',
#         'Q02083',
#         'P28325', 'P01135', 'P80075', 'P78423', 'P78556', 'P50225'
#     ],
#     None: [
#         'Q9HAN9',
#         'Q14108',
#         'P53634',
#         'P01135',
#         'P80511',
#         'P78423',
#         'P78556',
#         'P50225'],
# }
DATA_SUBSET_FEATURE_FILTERS = {
    'medulla': [],
    'csf': [],
    'cortex': [],
    None: [],
}
# drop sample col


def interpret_log_reg(clf, sfs, y, feature_names, save_file):
    equation = ' + '.join([f'{coef:.2f} * {name}' for coef,
                           name in zip(clf.coef_[0],
                                       feature_names[sfs.get_support()])])
    equation += f' + {clf.intercept_[0]:.2f}'
    # save to txt
    with open(f'{save_file}_kr.txt', 'w') as f:
        f.write(equation)


def interpret_tree(clf, X_sfs, y, feature_names, save_file):
    viz_model = dtreeviz.model(clf, X_sfs, y,
                               target_name='label',
                               feature_names=feature_names[sfs.get_support()],
                               class_names=['Control', 'Covid'])
    v = viz_model.view()
    v.save(f'{save_file}_tree.svg')


cross_validator = KFold(10, shuffle=True, random_state=11)

# calc setting combinations
GROUND_TRUTH = {}
PREDICTIONS = {}
for sfs_model, cls_model, feature_num, data_subset in product(
        SFS_MODELS, CLS_MODELS, FEATURE_NUMS, DATA_SUBSETS):
    print(
        f'Running {str(sfs_model)} {str(cls_model)} {feature_num} {data_subset}')
    data_to_use = data if data_subset is None else data[data['sample'] ==
                                                        data_subset]
    data_to_use = data_to_use.drop(columns=['sample'])
    features_to_drop = DATA_SUBSET_FEATURE_FILTERS[data_subset]
    data_to_use = data_to_use.drop(columns=features_to_drop)
    X = data_to_use.drop(columns=['label']).values
    y = data_to_use['label'].values
    feature_names = data_to_use.drop(columns=['label']).columns
    for train_index, test_index in cross_validator.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sfs = SequentialFeatureSelector(
            sfs_model(),
            n_features_to_select=feature_num,
            cv=5,
            scoring='neg_log_loss',
            n_jobs=4)
        sfs.fit(X_train, y_train)
        X_train_sfs = sfs.transform(X_train)
        X_test_sfs = sfs.transform(X_test)
        clf = cls_model(random_state=11)
        clf.fit(X_train_sfs, y_train)
        y_pred = clf.predict(X_test_sfs)
        y_pred_proba = clf.predict_proba(X_test_sfs)
        # save ground truth and predictions
        key = (sfs_model, cls_model, feature_num, data_subset)
        if key not in GROUND_TRUTH:
            GROUND_TRUTH[key] = []
            PREDICTIONS[key] = []
        GROUND_TRUTH[key].extend(y_test)
        PREDICTIONS[key].extend(y_pred_proba[:, 1])
    sfs = SequentialFeatureSelector(
        sfs_model(),
        n_features_to_select=feature_num,
        cv=5,
        scoring='neg_log_loss',
        n_jobs=4)
    sfs.fit(X, y)
    X_sfs = sfs.transform(X)
    clf = cls_model(random_state=11)
    clf.fit(X_sfs, y)
    # interpret
    if cls_model == LogisticRegression:
        # create dir
        cls_model_str = str(cls_model).split('.')[-1][:-2]
        sfs_model_str = str(sfs_model).split('.')[-1][:-2]
        if not os.path.exists(
                f'{ROOT_DIR}/{cls_model_str}/{data_subset}/'):
            os.makedirs(
                f'{ROOT_DIR}/{cls_model_str}/{data_subset}')
        explainer = shap.LinearExplainer(clf, X_sfs,)
        shap_values = explainer.shap_values(X_sfs, )
        shap_values = np.abs(shap_values)
        # group based on tissue
        # medulla
        medulla_indices = data[data['sample'] == 'medulla'].index
        shap_medulla = shap_values[medulla_indices]
        cortex_indices = data[data['sample'] == 'cortex'].index
        shap_cortex = shap_values[cortex_indices]
        csf_indices = data[data['sample'] == 'csf'].index
        shap_csf = shap_values[csf_indices]
        # grouped bar plot
        shap_medulla = np.mean(shap_medulla, axis=0)
        shap_cortex = np.mean(shap_cortex, axis=0)
        shap_csf = np.mean(shap_csf, axis=0)
        shap_order = np.argsort(np.mean(shap_values, axis=0))[::-1]
        shap_medulla = shap_medulla[shap_order]
        shap_cortex = shap_cortex[shap_order]
        shap_csf = shap_csf[shap_order]
        feature_names_temp = feature_names[sfs.get_support()]
        feature_names_temp = feature_names_temp[shap_order]
        shap_df = pd.DataFrame(
            {
                'Uniprot': feature_names_temp,
                'medulla': shap_medulla,
                'cortex': shap_cortex,
                'csf': shap_csf
            })
        shap_df.to_csv(
            f'{ROOT_DIR}/{cls_model_str}/{data_subset}/{sfs_model_str}_{feature_num}_all_shap_tissue.csv')
        shap_df = shap_df.melt(
            id_vars='Uniprot',
            var_name='Tissue',
            value_name='value')
        shap_df.to_csv(
            f'{ROOT_DIR}/{cls_model_str}/{data_subset}/{sfs_model_str}_{feature_num}_all_shap_tissue_melt.csv')
        import seaborn as sns
        plt.clf()
        sns.barplot(data=shap_df, x='Uniprot', y='value', hue='Tissue')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            f'{ROOT_DIR}/{cls_model_str}/{data_subset}/{sfs_model_str}_{feature_num}_all_shap_tissue.png')
        interpret_log_reg(
            clf, sfs, y, feature_names,
            f'{ROOT_DIR}/{cls_model_str}/{data_subset}/{sfs_model_str}_{feature_num}_all')
    elif cls_model == DecisionTreeClassifier:
        # create dir
        cls_model_str = str(cls_model).split('.')[-1][:-2]
        sfs_model_str = str(sfs_model).split('.')[-1][:-2]
        explainer = shap.TreeExplainer(clf, X_sfs,)
        shap_values = explainer.shap_values(X_sfs, )
        shap_values = shap_values[1]
        shap_values = np.abs(shap_values)
        # group based on tissue
        # medulla
        medulla_indices = data[data['sample'] == 'medulla'].index
        shap_medulla = shap_values[medulla_indices]
        cortex_indices = data[data['sample'] == 'cortex'].index
        shap_cortex = shap_values[cortex_indices]
        csf_indices = data[data['sample'] == 'csf'].index
        shap_csf = shap_values[csf_indices]
        # grouped bar plot
        shap_medulla = np.mean(shap_medulla, axis=0)
        shap_cortex = np.mean(shap_cortex, axis=0)
        shap_csf = np.mean(shap_csf, axis=0)
        shap_order = np.argsort(np.mean(shap_values, axis=0))[::-1]
        shap_medulla = shap_medulla[shap_order]
        shap_cortex = shap_cortex[shap_order]
        shap_csf = shap_csf[shap_order]
        feature_names_temp = feature_names[sfs.get_support()]
        feature_names_temp = feature_names_temp[shap_order]
        shap_df = pd.DataFrame(
            {
                'Uniprot': feature_names_temp,
                'medulla': shap_medulla,
                'cortex': shap_cortex,
                'csf': shap_csf
            })
        shap_df = shap_df.melt(
            id_vars='Uniprot',
            var_name='Tissue',
            value_name='value')
        import seaborn as sns
        plt.clf()
        sns.barplot(data=shap_df, x='Uniprot', y='value', hue='Tissue')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(
            f'{ROOT_DIR}/{cls_model_str}/{data_subset}/{sfs_model_str}_{feature_num}_all_shap_tissue.png')
        if not os.path.exists(
                f'{ROOT_DIR}/{cls_model_str}/{data_subset}/'):
            os.makedirs(
                f'{ROOT_DIR}/{cls_model_str}/{data_subset}')
        interpret_tree(
            clf,
            X_sfs,
            y,
            feature_names,
            f'{ROOT_DIR}/{cls_model_str}/{data_subset}/{sfs_model_str}_{feature_num}_all')
# calc accuracies
# save to json
SCORES = {}
for key in GROUND_TRUTH:
    accuracy = accuracy_score(
        GROUND_TRUTH[key],
        [1 if p > 0.5 else 0 for p in PREDICTIONS[key]])
    log_loss_score = log_loss(GROUND_TRUTH[key], PREDICTIONS[key])
    SCORES[key] = {'accuracy': accuracy, 'log_loss': log_loss_score}

# convert non-serializable objects to strings
for key in SCORES:
    for k in SCORES[key]:
        if isinstance(SCORES[key][k], np.float64):
            SCORES[key][k] = float(SCORES[key][k])
        elif isinstance(SCORES[key][k], np.int64):
            SCORES[key][k] = int(SCORES[key][k])
# convert tuple keys to strings
SCORES = {str(k): v for k, v in SCORES.items()}
with open(f'{ROOT_DIR}/scores.json', 'w') as f:
    json.dump(SCORES, f, indent=4)
