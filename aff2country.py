import time
import os
import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import joblib
import fire

from settings import *


def create_or_get_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def my_read_csv(fname, **kws):
    # use "na_filter=False" is to avoid reading "NA" (Namibia) as an empty value
    return pd.read_csv(fname, na_filter=False, **kws)

def print_list_nicely(lst, items_per_line=10, sep=' '):
    for i in range(0, len(lst), items_per_line):
        print('    ', *lst[i:i + items_per_line], sep=sep)

print(f'\n- DEBUG = {DEBUG}')
print(f'\n- MODEL_DIR = {MODEL_DIR}')
print(f'- DEFAULT_FOLD_FOR_CALIBRATED_CLASSIFIER = {DEFAULT_FOLD_FOR_CALIBRATED_CLASSIFIER}')
print(f'- COUNTRIES_OF_INTEREST: {len(COUNTRIES_OF_INTEREST)} countries\n')
print_list_nicely(COUNTRIES_OF_INTEREST, items_per_line=10)
print('\n*** To change the above settings, edit the file "settings.py" ***\n')


class Aff2Country:
    def __init__(self):
        self.working_dir = 'working'
        self.model_dir = MODEL_DIR
        self.debug = DEBUG
        self.cv_fold = DEFAULT_FOLD_FOR_CALIBRATED_CLASSIFIER

    def get_model_folder(self):
        return create_or_get_folder(self.model_dir)

    def get_working_folder(self):
        return create_or_get_folder(self.working_dir)

    def get_pkl_of_model_linearsvc(self):
        return f'{self.get_model_folder()}/model_linearsvc.pkl'

    def get_pkl_of_label_encoder(self):
        return f'{self.get_model_folder()}/label_encoder.pkl'

    def get_pkl_of_tfidf_vectorizer(self):
        return f'{self.get_model_folder()}/tfidf_vectorizer.pkl'

    def get_fpath_of_prediction(self, input):
        file_out = input.replace('.csv', '.pred.csv')
        print(f'\n- prediction output will be saved to: {file_out}\n')
        return file_out


    def train_model(self, input, ngram_range=(1, 2), min_df=2):
        if self.debug:
            nrows = 100000
            print(f'\n- only 100,000 rows in "{input}" will be used for training')
        else:
            nrows = None
            print('\n- this may take one hour to train on all 2.7 million data points\n')

        fpath = input
        if not os.path.exists(fpath):
            print('\n- training data file not found:', fpath)
            sys.exit()

        print('- input training file:', fpath)
        print('- model folder:', self.get_model_folder())

        df = my_read_csv(fpath, nrows=nrows)
        print()
        print(df[:3])
        print()

        df['country'] = df['country'].apply(lambda x: x if x in COUNTRIES_OF_INTEREST else '--')
        df_tmp = df.groupby('country').agg(cnt=('country', 'count')).sort_values('cnt', ascending=False).reset_index()
        print(f'\n- countries in the training data: {len(df_tmp)}')
        print('  "--" represents those countries out of our interest\n')
        print(df_tmp)
        print()

        df['text'] = df['org'] + " , " + df['gpe']

        df = df.sample(frac=1, random_state=0)
        X_train = df.text
        y_train = df.country

        le = preprocessing.LabelEncoder()
        le.fit(df.country.tolist())

        joblib.dump(le, self.get_pkl_of_label_encoder())
        print('- the label encoder will be saved to:', self.get_pkl_of_label_encoder())

        y_train = le.transform(y_train)

        clf = CalibratedClassifierCV(LinearSVC(C=1), method='sigmoid', cv=self.cv_fold)
        vectorizer = TfidfVectorizer
        vec = vectorizer(ngram_range=ngram_range, min_df=min_df)

        X_train_sparse = vec.fit_transform(X_train).astype(float)

        joblib.dump(vec, self.get_pkl_of_tfidf_vectorizer())
        print('- the tfidf vectorizer will be saved to:', self.get_pkl_of_tfidf_vectorizer())

        print('\n- training (will take a while) ...')
        fpath_model = self.get_pkl_of_model_linearsvc()
        clf.fit(X_train_sparse, y_train)
        joblib.dump(clf, fpath_model)
        print('\n- the classification model was saved to:', fpath_model)


    def predict_country(self, input, use_ad_hoc_countries=True):
        fpath_out = self.get_fpath_of_prediction(input)

        if os.path.exists(fpath_out):
            print('\n- prediction file already exists:', fpath_out)
            return fpath_out

        df = my_read_csv(input)  # "affiliation" or "org, gpe" if NER'ed
        print('- num of affiliations to predict:', len(df))

        is_ner_format = 'org' in df and 'gpe' in df
        if is_ner_format:
            print("- you are using the data format with columns \"org\" and \"gpe\".\n")

        print('- loading label encoding information')
        le = joblib.load(self.get_pkl_of_label_encoder())

        def _predict_inside(df, clf):
            if 'affiliation' in df:
                df = df.rename(columns={'affiliation': 'text'})
            elif not 'text' in df:
                print('- error: not found columns "text", "affiliation", or "org, gpe"')
                sys.exit()

            X_test_sparse = vec.transform(df['text']).astype(float)
            y_proba = clf.predict_proba(X_test_sparse)

            y_pred = np.argmax(y_proba, axis=1)
            df_out = df[['text']].copy()
            df_out['confidence'] = np.max(y_proba, axis=1)
            df_out['winner'] = le.inverse_transform(y_pred)
            return df_out

        print('- loading tfidf vectorizer')
        vec = joblib.load(self.get_pkl_of_tfidf_vectorizer())
        print('- loading LinearSVC model')
        clf = joblib.load(self.get_pkl_of_model_linearsvc())

        df_out = _predict_inside(df, clf)

        if use_ad_hoc_countries:
            # You can update the following code for other places or countries of interest
            adhoc_countries = {'CN': {'taiwan': 'TW', 'hong kong': 'HK'}}
            def _adjust_adhoc_countries(x):
                if x.winner in adhoc_countries:
                    adhocs = adhoc_countries[x.winner]
                    for cue, code in adhocs.items():
                        if cue in x['text'].lower():
                            return code
                return x.winner

            for country, adhocs in adhoc_countries.items():
                idx = df_out['winner'] == country
                df_out.loc[idx, 'winner'] = df_out.loc[idx].apply(_adjust_adhoc_countries, axis=1)

        print()
        print(df_out.round(2)[:3])
        print()
        print(df_out.iloc[0])

        df_out.to_csv(fpath_out, index=False, float_format="%.3f")
        print('\n- prediction results saved to:', fpath_out)
        return fpath_out


#--------------------
#
def run(task, input):
    a2c = Aff2Country()
    if task == 'train':
        a2c.train_model(input)
    elif task == 'infer':
        a2c.predict_country(input)
    else:
        print('\n- error: unknown task:', task)


if __name__ == '__main__':
    tic = time.time()
    fire.Fire(run)
    print(f'\n- time used: {time.time() - tic:.1f} seconds\n')