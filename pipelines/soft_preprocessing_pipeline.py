import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
set_config(transform_output="pandas")

class ShiftedLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.min_shifts_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            min_val = X[col].min()
            self.min_shifts_[col] = abs(min_val) if min_val < 0 else 0.0
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            shift = self.min_shifts_.get(col, 0.0)
            val_shifted = (X[col] + shift).clip(lower=0)
            X[col] = np.log1p(val_shifted)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features


def clean_strings_func(X):
    X = X.copy()
    obj_cols = X.select_dtypes(include='object').columns
    for col in obj_cols:
        X[col] = X[col].str.replace("'", "", regex=False).str.strip()
    return X

def engineer_features_func(X):
    X = X.copy()
    
    X['is_mig_universe'] = 0
    X['is_child'] = 0
    X['is_family_match'] = 0
    X['is_investor'] = 0
    
    if 'capital_gains' in X.columns:
        X['capital_gains'] = X['capital_gains'].clip(upper=99999)
    if 'wage_per_hour' in X.columns:
        X['wage_per_hour'] = X['wage_per_hour'].clip(upper=9999)
    
    mig_cols = ['mig_chg_reg', 'mig_move_reg', 'mig_chg_msa']
    if set(mig_cols).issubset(X.columns):
        c1 = X['mig_chg_reg'] == 'Not in universe'
        c2 = X['mig_move_reg'] == 'Not in universe'
        c3 = X['mig_chg_msa'] == 'Not in universe'
        X['is_mig_universe'] = (c1 & c2 & c3).astype(int)

    if 'fam_under_18' in X.columns:
        X['is_child'] = (X['fam_under_18'] != 'Not in universe').astype(int)

    fam_cols = ['country_father', 'country_mother', 'country_self']
    if set(fam_cols).issubset(X.columns):
        X['is_family_match'] = (
            (X['country_father'] == X['country_mother']) & 
            (X['country_mother'] == X['country_self'])
        ).astype(int)

    if 'capital_gains' in X.columns and 'capital_losses' in X.columns:
        X['net_capital'] = X['capital_gains'] - X['capital_losses']
        X['is_investor'] = ((X['capital_gains'] > 0) | (X['capital_losses'] > 0)).astype(int)
    else:
        X['net_capital'] = 0
    
    cols_to_drop = [
        'mig_chg_reg', 'mig_move_reg', 'mig_chg_msa', 'fam_under_18',
        'country_father', 'country_mother', 'country_self', 
        'det_ind_code', 'det_occ_code'
    ]
    return X.drop(columns=cols_to_drop, errors='ignore')

def group_categories_func(X):
    X = X.copy()
    
    if 'full_or_part_emp' in X.columns:
        def map_schedule(val):
            if val == "Full-time schedules": return 'Full-Time'
            if "PT" in val: return 'Part-Time'
            if val in ["Not in labor force", "Unemployed full-time", "Unemployed part-time"]: return 'Inactive'
            return 'Children/Other'
        X['schedule_group'] = X['full_or_part_emp'].map(map_schedule)
    else:
        X['schedule_group'] = 'Unknown' 

    if 'class_worker' in X.columns:
        class_map = {k: 'Non-Active' for k in ["Never worked", "Without pay", "Not in universe"]}
        class_map.update({k: 'State-Local-Gov' for k in ["Local government", "State government"]})
        X['class_group'] = X['class_worker'].replace(class_map)
    else:
        X['class_group'] = 'Unknown'

    hh_col = 'det_hh_fam_stat'
    if hh_col in X.columns:
        def map_hh(val):
            if "Householder" in val: return 'Householder'
            if "Spouse of householder" in val: return 'Spouse'
            return 'Dependent'
        X['household_role'] = X[hh_col].map(map_hh)
    else:
        X['household_role'] = 'Unknown'

    if 'citizenship' in X.columns:
        citizen_map = {
            'Foreign born- U S citizen by naturalization': 'Naturalized',
            'Foreign born- Not a citizen of U S': 'Non-Citizen'
        }
        X['citizenship_group'] = X['citizenship'].map(citizen_map).fillna('US Native')
    else:
        X['citizenship_group'] = 'US Native'

    cols_drop = ['full_or_part_emp', 'class_worker', 'det_hh_fam_stat', 'citizenship']
    return X.drop(columns=cols_drop, errors='ignore')


education_order = [
    'Children', 'Less than 1st grade', '1st 2nd 3rd or 4th grade', '5th or 6th grade',
    '7th and 8th grade', '9th grade', '10th grade', '11th grade', '12th grade no diploma',
    'High school graduate', 'Some college but no degree', 'Associates degree-occup /vocational',
    'Associates degree-academic program', 'Bachelors degree(BA AB BS)',
    'Masters degree(MA MS MEng MEd MSW MBA)', 'Prof school degree (MD DDS DVM LLB JD)',
    'Doctorate degree(PhD EdD)'
]

log_features = ['wage_per_hour', 'stock_dividends', 'net_capital', 'capital_gains', 'capital_losses']

continuous_features = ['age', 'num_emp', 'weeks_worked']
binary_features = ['is_mig_universe', 'is_child', 'is_family_match', 'is_investor']

ohe_features = [
    'hs_college', 'marital_stat', 'union_member',  'tax_filer_stat', 'mig_same',           
     
    'major_ind_code', 'major_occ_code', 'schedule_group',
    
     'class_group', 'household_role', 'citizenship_group' ,
]

preprocessor = ColumnTransformer(
    transformers=[
        ('ord_scale_edu', Pipeline([
            ('ordinal', OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=-1)),
            ('scale', StandardScaler())
        ]), ['education']),
        
        ('log_scale', Pipeline([
            ('log', ShiftedLogTransformer()),
            ('scale', StandardScaler())
        ]), log_features),

        ('num_scale', StandardScaler(), continuous_features),
        ('binary_pass', 'passthrough', binary_features),
        
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False, min_frequency=0.01), ohe_features)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

soft_preprocessing_pipeline = Pipeline([
    ('clean_strings', FunctionTransformer(clean_strings_func)),
    ('engineer_features', FunctionTransformer(engineer_features_func)),
    ('group_categories', FunctionTransformer(group_categories_func)),
    ('preprocessing', preprocessor)
])