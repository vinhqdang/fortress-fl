"""
Data Loader for Real-World Datasets

Utilities to load and preprocess real-world datasets for FORTRESS-FL evaluation.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

def load_credit_card_data(n_operators: int, test_size: float = 0.2,
                         byzantine_operators: List[int] = None) -> Tuple[List[Dict], Dict, int]:
    """
    Load and partition the 'default-of-credit-card-clients' dataset.
    
    Dataset: Default of Credit Card Clients
    Source: OpenML (ID: 42477)
    Task: Binary Classification (Default vs Non-Default)
    Features: 23 (Limit_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0..6, BILL_AMT1..6, PAY_AMT1..6)
    Samples: 30,000
    
    Args:
        n_operators: Number of operators to partition data among
        test_size: Fraction of data to use for testing
        byzantine_operators: List of Byzantine operator indices
        
    Returns:
        operators_data: List of operator data dicts
        test_data: Dict {'X': X_test, 'y': y_test}
        model_dim: Number of features
    """
    print("Downloading 'default-of-credit-card-clients' dataset from OpenML...")
    # Fetch dataset (ID 42477 is the standard version)
    try:
        data = fetch_openml(data_id=42477, as_frame=False, parser='auto')
    except Exception as e:
        print(f"Error fetching from OpenML: {e}")
        print("Falling back to synthetic data generation...")
        return None, None, 0

    X, y = data.data, data.target
    
    # Encode labels (0/1)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Add bias term (intercept)
    X = np.c_[X, np.ones(X.shape[0])]
    model_dim = X.shape[1]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Partition data among operators (IID for simplicity, or non-IID if needed)
    # Here we use random shuffling for IID partition
    n_train = len(X_train)
    indices = np.random.permutation(n_train)
    partition_size = n_train // n_operators
    
    operators_data = []
    if byzantine_operators is None:
        byzantine_operators = []
        
    for i in range(n_operators):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size
        op_indices = indices[start_idx:end_idx]
        
        op_X = X_train[op_indices]
        op_y = y_train[op_indices]
        
        operator_data = {
            'id': f'Operator_{i}',
            'dataset': {'X': op_X, 'y': op_y},
            'is_byzantine': i in byzantine_operators,
            'attack_type': 'sign_flip' if i in byzantine_operators else None
        }
        operators_data.append(operator_data)
        
    test_data = {'X': X_test, 'y': y_test}
    
    print(f"Loaded Credit Card dataset: {n_train} train samples, {len(y_test)} test samples")
    print(f"Features: {model_dim} (including bias)")
    print(f"Partitioned among {n_operators} operators (~{partition_size} samples each)")
    
    return operators_data, test_data, model_dim
