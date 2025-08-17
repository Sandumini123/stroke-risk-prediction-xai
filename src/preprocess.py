import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

def load_dataset(file_path):
    """Load and perform basic validation of the dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Target distribution:\n{df['stroke'].value_counts()}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def impute_missing_values(df):
    """Handle missing values with appropriate strategies"""
    print("\nHandling missing values...")
    missing_info = df.isnull().sum()
    print(f"Missing values before imputation:\n{missing_info[missing_info > 0]}")

    if 'bmi' in df.columns and df['bmi'].isnull().sum() > 0:
        df['bmi'] = df.groupby(['gender', pd.cut(df['age'], bins=[0, 30, 50, 70, 100])])['bmi'].transform(
            lambda x: x.fillna(x.median())
        )
        df['bmi'].fillna(df['bmi'].median(), inplace=True)

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"Missing values after imputation:\n{df.isnull().sum().sum()}")
    return df

def encode_features(df):
    """Encode categorical features"""
    print("\nEncoding categorical features...")
    
    if 'gender' in df.columns:
        df['gender_encoded'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 0.5})

    if 'ever_married' in df.columns:
        df['ever_married_encoded'] = df['ever_married'].map({'Yes': 1, 'No': 0})

    if 'work_type' in df.columns:
        work_type_map = {
            'Private': 3, 'Self-employed': 2, 'Govt_job': 1,
            'children': 0, 'Never_worked': 0
        }
        df['work_type_encoded'] = df['work_type'].map(work_type_map)

    if 'Residence_type' in df.columns:
        df['residence_encoded'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

    if 'smoking_status' in df.columns:
        smoking_map = {
            'formerly smoked': 2, 'smokes': 3, 'never smoked': 1, 'Unknown': 0
        }
        df['smoking_status_encoded'] = df['smoking_status'].map(smoking_map)

    cols_to_drop = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    print(f"Features after encoding: {list(df.columns)}")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    print(f"\nSplitting data - Features: {X.shape}, Target: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test

def handle_class_imbalance(X_train, y_train, method='adasyn', random_state=42):
    """Handle class imbalance using various resampling techniques"""
    print(f"\nHandling class imbalance using {method}...")
    print(f"Original class distribution: {Counter(y_train)}")

    try:
        if method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=random_state, n_neighbors=3)
        elif method == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=random_state)
        elif method == 'borderline_smote':
            from imblearn.over_sampling import BorderlineSMOTE
            sampler = BorderlineSMOTE(random_state=random_state)
        else:
            print(f"Unknown method {method}, using ADASYN")
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=random_state, n_neighbors=3)

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"Resampled class distribution: {Counter(y_resampled)}")
        return X_resampled, y_resampled

    except Exception as e:
        print(f"Error in resampling: {e}")
        print("Returning original data...")
        return X_train, y_train