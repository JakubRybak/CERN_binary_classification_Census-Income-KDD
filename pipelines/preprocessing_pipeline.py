import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


def clean_strings_func(X):
    X = X.copy()
    obj_cols = X.select_dtypes(include='object').columns
    for col in obj_cols:
        X[col] = X[col].str.replace("'", "", regex=False).str.strip()
    return X

def engineer_features_func(X):
    X = X.copy()
    X['capital_gains'] = X['capital_gains'].clip(upper=99999)
    
    if 'wage_per_hour' in X.columns:
        X['wage_per_hour'] = X['wage_per_hour'].clip(upper=9999)

    X['is_mig_universe'] = (
        (X['mig_chg_reg'] == 'Not in universe') & 
        (X['mig_move_reg'] == 'Not in universe') & 
        (X['mig_chg_msa'] == 'Not in universe')
    ).astype(int)

    X['is_child'] = (X['fam_under_18'] != 'Not in universe').astype(int)

    X['is_family_match'] = (
        (X['country_father'] == X['country_mother']) & 
        (X['country_mother'] == X['country_self'])
    ).astype(int)

    X['net_capital'] = X['capital_gains'] - X['capital_losses']
    X['is_investor'] = ((X['capital_gains'] > 0) | (X['capital_losses'] > 0)).astype(int)
    
    cols_to_drop = [
        'mig_chg_reg', 'mig_move_reg', 'mig_chg_msa', 'fam_under_18',
        'country_father', 'country_mother', 'country_self',
        'capital_gains', 'capital_losses', 'own_or_self', 'sex', 'race', 'hisp_origin'
    ]
    return X.drop(columns=cols_to_drop, errors='ignore')

def group_categories_func(X):
    X = X.copy()
    
    industry_map = {
        "Mining": 'High_Yield', "Finance insurance and real estate": 'High_Yield', "Public administration": 'High_Yield', "Communications": 'High_Yield', "Manufacturing-durable goods": 'High_Yield',
        "Retail trade": 'Low_Yield', "Personal services except private HH": 'Low_Yield', "Agriculture": 'Low_Yield', "Private household services": 'Low_Yield', "Wholesale trade": 'Low_Yield', "Forestry and fisheries": 'Low_Yield', "Entertainment": 'Low_Yield'
    }
    
    occ_map = {
        "Executive admin and managerial": 'High_Income', "Professional specialty": 'High_Income',
        "Sales": 'Mid_Income', "Protective services": 'Mid_Income', "Technicians and related support": 'Mid_Income', "Precision production craft & repair": 'Mid_Income',
        "Machine operators assmblrs & inspctrs": 'Low_Income', "Transportation and material moving": 'Low_Income', "Farming forestry and fishing": 'Low_Income', "Handlers equip cleaners etc": 'Low_Income', "Adm support including clerical": 'Low_Income'
    }
    
    X['industry_group'] = X['major_ind_code'].map(industry_map).fillna('Average_Yield')
    X['occupation_group'] = X['major_occ_code'].map(occ_map).fillna('Minimal_Income')
    
    def map_schedule(val):
        if val == "Full-time schedules": return 'Full-Time'
        if "PT" in val: return 'Part-Time'
        if val in ["Not in labor force", "Unemployed full-time", "Unemployed part-time"]: return 'Inactive'
        return 'Children/Other'
    X['schedule_group'] = X['full_or_part_emp'].map(map_schedule)

    class_map = {k: 'Non-Active' for k in ["Never worked", "Without pay", "Not in universe"]}
    class_map.update({k: 'State-Local-Gov' for k in ["Local government", "State government"]})

    X['class_group'] = X['class_worker'].replace(class_map)

    def map_hh(val):
        if "Householder" in val: return 'Householder'
        if "Spouse of householder" in val: return 'Spouse'
        return 'Dependent'
    X['household_role'] = X['det_hh_fam_stat'].map(map_hh)

    citizen_map = {
        'Foreign born- U S citizen by naturalization': 'Naturalized',
        'Foreign born- Not a citizen of U S': 'Non-Citizen'
    }
    X['citizenship_group'] = X['citizenship'].map(citizen_map).fillna('US Native')

    cols_drop = ['major_ind_code', 'major_occ_code', 'full_or_part_emp', 
                 'class_worker', 'det_hh_fam_stat', 'citizenship']
    return X.drop(columns=cols_drop, errors='ignore')

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
        if input_features is not None:
            return input_features
        
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
            
        raise ValueError("Could not determine feature names. Ensure fit() was called.")


education_order = [
    'Children', 'Less than 1st grade', '1st 2nd 3rd or 4th grade', '5th or 6th grade',
    '7th and 8th grade', '9th grade', '10th grade', '11th grade', '12th grade no diploma',
    'High school graduate', 'Some college but no degree', 'Associates degree-occup /vocational',
    'Associates degree-academic program', 'Bachelors degree(BA AB BS)',
    'Masters degree(MA MS MEng MEd MSW MBA)', 'Prof school degree (MD DDS DVM LLB JD)',
    'Doctorate degree(PhD EdD)'
]

log_features = ['wage_per_hour', 'stock_dividends', 'net_capital']

continuous_features = ['age', 'num_emp', 'weeks_worked']
binary_features = ['is_mig_universe', 'is_child', 'is_family_match', 'is_investor']

ohe_features = [
    'hs_college', 'marital_stat', 'union_member', 'tax_filer_stat', 'mig_same',
    'industry_group', 'occupation_group', 'schedule_group', 
    'class_group', 'household_role', 'citizenship_group'
]


preprocessor = ColumnTransformer(
    transformers=[
        ('ord_scale_edu', Pipeline([
            ('ordinal', OrdinalEncoder(categories=  [education_order], handle_unknown='use_encoded_value', unknown_value=-1)),
            ('scale', StandardScaler())
        ]), ['education']),
        
        ('log_scale', Pipeline([
            ('log', ShiftedLogTransformer()),
            ('scale', StandardScaler())
        ]), log_features),

        ('num_scale', StandardScaler(), continuous_features),
        ('binary_pass', 'passthrough', binary_features),
        
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ohe_features)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)


preprocessing_pipeline = Pipeline([
    ('clean_strings', FunctionTransformer(clean_strings_func)),
    ('engineer_features', FunctionTransformer(engineer_features_func)),
    ('group_categories', FunctionTransformer(group_categories_func)),
    ('preprocessing', preprocessor)
])