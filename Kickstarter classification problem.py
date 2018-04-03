from __future__ import division
import math
from datetime import datetime
import json
import pickle  # do not remove this
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = pd.read_csv('data\kick_train.csv', index_col='Unnamed: 0')
data.drop(['staff_pick', 'backers_count'], axis=1, inplace=True)
data['state'] = [1 if x == 'successful' else 0 for x in data['state']]
#################################################################
xtrain, xtest, ytrain, ytest = train_test_split(data.drop('state', axis=1), data['state'],
                                                test_size=0.3, stratify=data['state'], random_state=456)
print(sum(ytrain) / ytrain.shape[0])
print(sum(ytest) / ytest.shape[0])


#################################################################


def unix_date_diff(df, col1, col2):
    out_list = []
    for x, y in zip(df[col1], df[col2]):
        z = math.floor(abs(x / 86400 - y / 86400))
        out_list.append(z)
    return out_list


def one_hot_encoding(df, val_dict):
    model_df = pd.DataFrame()
    for col in df.columns:
        dummy_df = pd.get_dummies(df[col].astype('category', categories=val_dict[col]), prefix=col)
        model_df = pd.concat([model_df, dummy_df], axis=1)
    return model_df


def categories_ohe(df):
    rcat_df = df[new_cat_features]
    cat_dict = {}
    for col in rcat_df.columns:
        rcategories = rcat_df[col].unique()
        cat_dict[col] = rcategories
    return cat_dict


def preprocess_data(df):
    df.loc[:, "profile_name"] = df["profile"].apply(lambda x: json.loads(x).get("name", None))
    df.loc[:, "profile_name"] = df["profile_name"].apply(lambda x: 1 if x == 'null' else 0)
    df.loc[:, "profile_blurb"] = df["profile"].apply(lambda x: json.loads(x).get("blurb", None))
    df.loc[:, "profile_blurb"] = df["profile_blurb"].apply(lambda x: 1 if x == 'null' else 0)
    df.loc[:, "link"] = df["profile"].apply(lambda x: json.loads(x).get("link_url", None))
    df.loc[:, "link"] = df["link"].apply(lambda x: 1 if x == 'null' else 0)
    df.loc[:, "should_show_feature_image_section"] = df["profile"].apply(lambda x: json.loads(x).get(
        "should_show_feature_image_section", None))
    df.loc[:, "show_feature_image"] = df["profile"].apply(lambda x: json.loads(x).get("show_feature_image", None))
    df.loc[:, "new_id"] = df["category"].apply(lambda x: json.loads(x).get("id", None))
    df.loc[:, "position"] = df["category"].apply(lambda x: json.loads(x).get("position", None))
    df.loc[:, "colour"] = df["category"].apply(lambda x: json.loads(x).get("color", None))
    df.loc[:, "new_category"] = df["category"].apply(lambda x: json.loads(x).get("parent_id", None))
    df.loc[:, "sub_category"] = df["category"].apply(lambda x: json.loads(x).get("name", None))
    df.loc[:, "launched_month"] = df["launched_at"].apply(lambda x: (datetime.fromtimestamp(
        int(x))).strftime('%Y-%m-%d %H:%M:%S'))
    df.loc[:, "launched_month"] = [(x.date()).month for x in pd.to_datetime(df["launched_month"])]
    df.loc[:, "created_month"] = df["created_at"].apply(lambda x: (datetime.fromtimestamp(
        int(x))).strftime('%Y-%m-%d %H:%M:%S'))
    df.loc[:, "created_month"] = [(x.date()).month for x in pd.to_datetime(df["created_month"])]
    df.loc[:, "deadline_month"] = df["deadline"].apply(lambda x: (datetime.fromtimestamp(
        int(x))).strftime('%Y-%m-%d %H:%M:%S'))
    df.loc[:, "deadline_month"] = [(x.date()).month for x in pd.to_datetime(df["deadline_month"])]
    df.loc[:, "launched_wk"] = df["launched_at"].apply(lambda x: (datetime.fromtimestamp(
        int(x))).strftime('%Y-%m-%d %H:%M:%S'))
    df.loc[:, "launched_wk"] = [(x.date()).weekday() for x in pd.to_datetime(df["launched_wk"])]
    df.loc[:, "created_wk"] = df["created_at"].apply(lambda x: (datetime.fromtimestamp(
        int(x))).strftime('%Y-%m-%d %H:%M:%S'))
    df.loc[:, "created_wk"] = [(x.date()).weekday() for x in pd.to_datetime(df["created_wk"])]
    df.loc[:, "deadline_wk"] = df["deadline"].apply(lambda x: (datetime.fromtimestamp(
        int(x))).strftime('%Y-%m-%d %H:%M:%S'))
    df.loc[:, "deadline_wk"] = [(x.date()).weekday() for x in pd.to_datetime(df["deadline_wk"])]
    df.loc[:, 'new_goal'] = df['goal'] * df['static_usd_rate']
    df.loc[:, "desc_word_count"] = df["blurb"].apply(lambda x: len(str(x).split()))
    df.loc[:, 'Create_Launch'] = unix_date_diff(df, 'launched_at', 'created_at')
    df.loc[:, 'Create_Deadline'] = unix_date_diff(df, 'deadline', 'created_at')
    df.loc[:, 'Launch_Deadline'] = unix_date_diff(df, 'deadline', 'launched_at')
    return df


def cat_transform(df, cat_dict):
    for cat in cat_dict.keys():
        new_dict_v = cat_dict[cat]
        df.loc[:, cat] = [new_dict_v[i] for i in df[cat]]
        return df


class Num(TransformerMixin, BaseEstimator):
    def fit(self, df):
        return df

    def transform(self, df):
        rev_num_df = df[new_num_features]
        rev_cat_df = df[new_cat_features]
        rev_cat_df = cat_transform(rev_cat_df, new_cat_dict)
        mod_df = one_hot_encoding(rev_cat_df, categories)
        return (pd.concat([rev_num_df, mod_df], axis=1)).values


class New_Feat(TransformerMixin, BaseEstimator):
    def fit(self, df):
        return df

    def transform(self, df):
        df = preprocess_data(df)
        return df[features]


proc_xtrain = preprocess_data(xtrain)
cols = proc_xtrain.columns
#################################################################
new_feat = list(cols[-21:])
features = ['disable_communication', 'country']
features.extend(new_feat)
features.extend(['static_usd_rate'])
print(len(features))
print(features)
#################################################################
cat_features = features[:-6]
num_features = features[-6:]
print(cat_features)
print(num_features)
#################################################################
new_num_features = []
for num in num_features:
    num_df = pd.concat([proc_xtrain[[num]], ytrain], axis=1)
    num_df1 = num_df[num_df['state'] == 1]
    suc_tot = np.percentile(num_df1[num], 50)
    num_df2 = num_df[num_df['state'] == 0]
    unsuc_tot = np.percentile(num_df2[num], 50)
    if suc_tot != unsuc_tot:
        new_num_features.append(num)
    print('Median {} of success to non-success is {:.2f}, {:.2f}'.format(num, suc_tot, unsuc_tot))
print(new_num_features)
#################################################################
for num in new_num_features:
    sns.distplot(proc_xtrain[[num]])
    plt.title(num)
    # plt.show()
#################################################################
new_cat_features = []
new_cat_dict = {}
for cat in cat_features:
    cat_df = proc_xtrain[cat]
    cat_df = cat_df.value_counts() / cat_df.shape[0]
    cat_df = cat_df.to_frame()
    cat_df = cat_df.reset_index()
    #     cat_df.loc[:, 'Cat'] = ['others' if x < 0.025 else y for x, y in
    #                             zip(cat_df[cat], cat_df[cat_df.columns[0]])]
    cat_df.loc[:, 'Cat'] = [y for y in cat_df[cat_df.columns[0]]]
    cat_df2 = cat_df.groupby('Cat')[cat].sum()
    cat_df2 = cat_df2.reset_index()
    # if min(cat_df2[cat]) > 0.025 and max(cat_df2[cat]) < 1:
    if max(cat_df2[cat]) < 1:
        new_cat_features.append(cat)
        new_cat_dict[cat] = cat_df.set_index('index')['Cat'].to_dict()
print('New category features are {}'.format(new_cat_features))
print('New category dict is {}'.format(new_cat_dict))
#################################################################
new_train_dataf = proc_xtrain[new_cat_features]
for cat in new_cat_dict.keys():
    new_dict = new_cat_dict[cat]
    new_train_dataf.loc[:, cat] = [new_dict[i] for i in new_train_dataf[cat]]
categories = categories_ohe(new_train_dataf)
print('New categories are {}'.format(categories))
#################################################################
pipeline1 = Pipeline(steps=[
    ('new_feature', New_Feat()),
    ('feature_preprocessing', Num())]
)
pipeline2 = Pipeline(steps=[
    ('best_feat', SelectKBest(score_func=chi2)),
    ('dtc', DecisionTreeClassifier())
]
)
#################################################################
xtrain2 = pipeline1.transform(xtrain)
xtest2 = pipeline1.transform(xtest)
print(xtrain2.shape)
print(xtest2.shape)
################################################################
chi_list = list(chi2(xtrain2, ytrain.values))
new_df = pd.DataFrame(data=chi_list)
chi_df = round(new_df.transpose(), 2)
chi = chi_df[chi_df[1] < 0.05].shape[0]
print(chi)
#################################################################
params = {'dtc__max_depth': np.arange(7, 15, 1),
          'dtc__criterion': ['gini', 'entropy'],
          'dtc__class_weight': [None, 'balanced'],
          'best_feat__k': [chi]
          }
#################################################################
gcv = GridSearchCV(pipeline2, params, cv=5)
xtrain2 = pipeline1.transform(xtrain)
gcv.fit(xtrain2, ytrain.values)
print(gcv.best_estimator_)
xtest2 = pipeline1.transform(xtest)
ypred2 = gcv.predict(xtest2)
print(classification_report(ytest, ypred2))
print(confusion_matrix(ytest, ypred2))
print(accuracy_score(ytest, ypred2))


