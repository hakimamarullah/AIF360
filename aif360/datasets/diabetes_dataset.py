import os

import pandas as pd

from aif360.datasets import StandardDataset

default_mappings = {
    'protected_attribute_maps': [
        {1.0: 'Emergency', 2.0: 'Urgent', 3.0: 'Elective', 4.0: 'Newborn',
         5.0: 'Not Available', 6.0: 'NULL', 7.0: 'Trauma Center', 8.0: 'Not Mapped'
         },
        {1.0: 'Discharged to home', 2.0: 'Discharged/transferred to another short term hospital', 3.0: 'Discharged/transferred to SNF',
         4.0: 'Discharged/transferred to ICF', 5.0: 'Discharged/transferred to another type of inpatient care institution', 6.0: 'Discharged/transferred to home with home health service', 7.0: 'Left AMA',
         8.0: 'Discharged/transferred to home under care of Home IV provider', 9.0: 'Admitted as an inpatient to this hospital',
         10.0: 'Neonate discharged to another hospital for neonatal aftercare', 11.0: 'Expired',
         12.0: 'Still patient or expected to return for outpatient services', 13.0: 'Hospice / home', 14.0: 'Hospice / medical facility',
         15.0: 'Discharged/transferred within this institution to Medicare approved swing bed', 16.0: 'Discharged/transferred/referred another institution for outpatient services',
         17.0: 'Discharged/transferred/referred to this institution for outpatient services', 18.0: 'NULL',
         19.0: 'Expired at home. Medicaid only, hospice.', 20.0: 'Expired in a medical facility. Medicaid only, hospice.',
         21.0: 'Expired, place unknown. Medicaid only, hospice.', 22.0: 'Discharged/transferred to another rehab fac including rehab units of a hospital .',
         23.0: 'Discharged/transferred to a long term care hospital.', 24.0: 'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
         25.0: 'Not Mapped', 26.0: 'Unknown/Invalid', 27.0: 'Discharged/transferred to a federal health care facility.',
         28.0: 'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
         29.0: 'Discharged/transferred to a Critical Access Hospital (CAH).', 30.0: 'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere'
         },
        {
            1.0: 'Physician Referral',
            2.0: 'Clinic Referral',
            3.0: 'HMO Referral',
            4.0: 'Transfer from a hospital',
            5.0: 'Transfer from a Skilled Nursing Facility (SNF)',
            6.0: 'Transfer from another health care facility',
            7.0: 'Emergency Room',
            8.0: 'Court/Law Enforcement',
            9.0: 'Not Available',
            10.0: 'Transfer from critial access hospital',
            11.0: 'Normal Delivery',
            12.0: 'Premature Delivery',
            13.0: 'Sick Baby',
            14.0: 'Extramural Birth',
            15.0: 'Not Available',
            17.0: 'NULL',
            18.0: 'Transfer From Another Home Health Agency',
            19.0: 'Readmission to Same Home Health Agency',
            20.0: 'Not Mapped',
            21.0: 'Unknown/Invalid',
            22.0: 'Transfer from hospital inpt/same fac reslt in a sep claim',
            23.0: 'Born inside this hospital',
            24.0: 'Born outside this hospital',
            25.0: 'Transfer from Ambulatory Surgery Center',
            26.0: 'Transfer from Hospice'
        }
    ]
}


class DiabetesDataset(StandardDataset):
    """Diabetes Dataset.

    See :file:`aif360/data/raw/diabetes/README.md`.
    """

    def __init__(self, label_name='readmitted ',
                 favorable_classes=['<30'],
                 protected_attribute_names=[
                     'admission_type_id', 'discharge_disposition_id', 'admission_source_id'],
                 privileged_classes=[],
                 instance_weights_name=None,
                 categorical_features=['race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
                                       'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin',
                                       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                                       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                                       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                                       'tolazamide', 'examide', 'citoglipton', 'insulin',
                                       'glyburide-metformin', 'glipizide-metformin',
                                       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                                       'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted'],
                 features_to_keep=[],
                 features_to_drop=['weight'],
                 na_values=['?'],
                 custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.
        """
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data', 'raw', 'diabetes', 'diabetic_data.csv')

        try:
            df = pd.read_csv(filepath, sep=';', na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print(
                "\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip")
            print("\nunzip it and place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
                os.path.abspath(__file__), '..', '..', 'data', 'raw', 'diabetes'))))
            import sys
            sys.exit(1)

        super(DiabetesDataset, self).__init__(df=df, label_name=label_name,
                                              favorable_classes=favorable_classes,
                                              protected_attribute_names=protected_attribute_names,
                                              privileged_classes=privileged_classes,
                                              instance_weights_name=instance_weights_name,
                                              categorical_features=categorical_features,
                                              features_to_keep=features_to_keep,
                                              features_to_drop=features_to_drop, na_values=na_values,
                                              custom_preprocessing=custom_preprocessing, metadata=metadata)
