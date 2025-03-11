import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import importlib.util
import sys
import os

# Load the student's code dynamically
spec = importlib.util.spec_from_file_location("starter_code", "starter_code.py")
starter = importlib.util.module_from_spec(spec)
sys.modules["starter_code"] = starter
spec.loader.exec_module(starter)

# Ensure the data file exists
data_path = 'data/credit_card_default.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure 'credit_card_default.csv' is in the 'data' folder.")

# Load the actual dataset (this will override the student's pd.read_csv call)
starter.ccd = pd.read_csv(data_path, index_col="ID")

class TestStarterCode(unittest.TestCase):

    def setUp(self):
        # Reload the module to reset state for each test with the actual data
        spec.loader.exec_module(starter)

    def test_data_loading(self):
        # Check if data is loaded and indexed by 'ID'
        self.assertTrue(isinstance(starter.ccd, pd.DataFrame), "ccd should be a DataFrame")
        self.assertEqual(starter.ccd.index.name, 'ID', "Index should be 'ID'")
        self.assertGreater(len(starter.ccd), 0, "Dataset should not be empty")

    def test_pay_feature_cleaning(self):
        # Check if PAY_ features are cleaned (no -1 or -2 values)
        pay_features = ['PAY_' + str(i) for i in range(1, 7)]
        for feature in pay_features:
            self.assertFalse((starter.ccd[feature] < 0).any(), f"{feature} should not contain negative values")

    def test_column_rename(self):
        # Check if 'default payment next month' is renamed to 'default'
        self.assertIn('default', starter.ccd.columns, "'default' column should exist")
        self.assertNotIn('default payment next month', starter.ccd.columns, "'default payment next month' should be renamed")

    def test_standardization(self):
        # Check if numerical features are standardized (mean ~0, std ~1)
        numerical_features = ['LIMIT_BAL', 'AGE'] + ['PAY_' + str(i) for i in range(1, 7)] + \
                            ['BILL_AMT' + str(i) for i in range(1, 7)] + ['PAY_AMT' + str(i) for i in range(1, 7)]
        for feature in numerical_features:
            mean = starter.ccd[feature].mean()
            std = starter.ccd[feature].std()
            self.assertAlmostEqual(mean, 0, delta=0.1, msg=f"{feature} mean should be close to 0")
            self.assertAlmostEqual(std, 1, delta=0.1, msg=f"{feature} std should be close to 1")

    def test_dummy_variables(self):
        # Check if categorical features are converted to dummy variables
        self.assertNotIn('SEX', starter.ccd.columns, "SEX should be converted to dummy variables")
        self.assertNotIn('EDUCATION', starter.ccd.columns, "EDUCATION should be converted to dummy variables")
        self.assertNotIn('MARRIAGE', starter.ccd.columns, "MARRIAGE should be converted to dummy variables")
        self.assertIn('SEX_2', starter.ccd.columns, "SEX_2 dummy variable should exist")
        self.assertIn('EDUCATION_2', starter.ccd.columns, "EDUCATION_2 dummy variable should exist")
        self.assertIn('MARRIAGE_2', starter.ccd.columns, "MARRIAGE_2 dummy variable should exist")

    def test_data_splitting(self):
        # Check if data is split correctly (assuming ~30,000 rows in the dataset)
        expected_train_size = int(0.8 * len(starter.ccd))
        expected_test_size = int(0.2 * len(starter.ccd))
        self.assertEqual(len(starter.X_train), expected_train_size, 
                         f"X_train should have 80% of the data (~{expected_train_size} rows)")
        self.assertEqual(len(starter.X_test), expected_test_size, 
                         f"X_test should have 20% of the data (~{expected_test_size} rows)")
        self.assertEqual(len(starter.y_train), expected_train_size, "y_train should match X_train length")
        self.assertEqual(len(starter.y_test), expected_test_size, "y_test should match X_test length")

    def test_models_defined(self):
        # Check if models are instantiated and trained
        for model_name in ['RandomForestClassifier', 'DecisionTreeClassifier', 'LogisticRegression']:
            self.assertTrue(any(isinstance(var, eval(model_name)) for var in vars(starter).values()),
                            f"{model_name} should be defined and instantiated")

    def test_performance_table(self):
        # Check if results_df exists and has the correct structure
        self.assertTrue(hasattr(starter, 'results_df'), "results_df should be defined")
        self.assertEqual(list(starter.results_df.columns), ['Model', 'Training Accuracy', 'Testing Accuracy', 'AUROC Score'],
                         "results_df should have correct column names")
        self.assertEqual(len(starter.results_df), 3, "results_df should have 3 rows (one per model)")
        self.assertEqual(sorted(starter.results_df['Model'].tolist()), 
                         ['Decision Tree', 'Logistic Regression', 'Random Forest'],
                         "results_df should include all three models")
        # Check if metrics are numeric (converted from string if formatted)
        for col in ['Training Accuracy', 'Testing Accuracy', 'AUROC Score']:
            try:
                starter.results_df[col].astype(float)
            except ValueError:
                self.fail(f"{col} values should be convertible to float")

if __name__ == '__main__':
    unittest.main()