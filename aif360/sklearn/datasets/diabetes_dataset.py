import os

import pandas as pd
import numpy as np

from aif360.sklearn.datasets.utils import standardize_dataset


# cache location
DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'raw')
DIABETES_URL = 'https://raw.githubusercontent.com/angelmanzur/Diabetes_130Hospitals/master/dataset_diabetes/diabetic_data.csv'


def fetch_diabetes(*, data_home=None, cache=True,
                   usecols=['patient_nbr', 'race', 'gender', 'age', 'admission_type_id',
                            'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
                            'medical_specialty', 'num_lab_procedures',
                            'num_procedures', 'num_medications', 'number_outpatient',
                            'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
                            'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',
                            'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                            'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                            'tolazamide', 'examide', 'citoglipton', 'insulin',
                            'glyburide-metformin', 'glipizide-metformin',
                            'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                            'metformin-pioglitazone', 'change', 'diabetesMed'],
                   dropcols=['weight', 'payer_code'], numeric_only=False, dropna=True):
    """Load the Diabetes dataset.
    
    Args:
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
        cache (bool): Whether to cache downloaded datasets.
        binary_race (bool, optional): Filter only White and Black defendants.
        usecols (single label or list-like, optional): Feature column(s) to
            keep. All others are dropped.
        dropcols (single label or list-like, optional): Feature column(s) to
            drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.
    """
    data_url = DIABETES_URL
    cache_path = os.path.join(data_home or DATA_HOME_DEFAULT,
                              os.path.basename(data_url))
    if cache and os.path.isfile(cache_path):
        df = pd.read_csv(cache_path, index_col='encounter_id')
    else:
        df = pd.read_csv(data_url, index_col='encounter_id')
        if cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)

    #remap target
    df = df.replace(
        {'NO':'1', '<30':'2', '>30':'3'})

    # remap null values to np.NaN instead of '?'
    df = df.replace({'?': np.NaN})

    # remap column age
    df = df.replace({'[0-10)' : '1', '[10-20)':'2', '[20-30)':'3', '[30-40)':'4', '[40-50)':'5',
    '[50-60)':'6', '[60-70)' :'7', '[70-80)' :'8', '[80-90)':'9', '[90-100)':'10'})
    
    # fill na
    for col in df.select_dtypes(include=object):
        df[col] = df[col].fillna(df[col].mode()[0])

    return standardize_dataset(df, prot_attr=['admission_type_id', 'discharge_disposition_id', 'admission_source_id'],
                               target='readmitted', usecols=usecols,
                               dropcols=dropcols, numeric_only=numeric_only,
                               dropna=dropna)
