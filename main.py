import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('payment_dataset.csv')
print("Data head:")
print(data.head())
print(f"\nDataset shape: {data.shape}")

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum()) 

# Check transaction types distribution
print("\nTransaction types:")
print(data.type.value_counts())

# =============================================================================
# EXPLICIT FRAUD DETECTION RULES
# =============================================================================

def detect_fraud_rules(row):
    """
    Apply explicit business rules to detect suspicious transactions
    """
    suspicious_flags = []
    
    # Rule 1: Balance inconsistency (mathematical impossibility)
    expected_new_balance = row['oldbalanceOrg'] - row['amount']
    if abs(expected_new_balance - row['newbalanceOrig']) > 0.01:  # Small tolerance for floating point
        suspicious_flags.append("Balance_Inconsistency")
    
    # Rule 2: Impossible transaction (spending more than available)
    if row['amount'] > row['oldbalanceOrg'] and row['oldbalanceOrg'] > 0:
        # Only flag if no overdraft protection (newbalance should be negative)
        if row['newbalanceOrig'] >= 0:
            suspicious_flags.append("Impossible_Transaction")
    
    # Rule 3: Large cash out operations
    if row['type'] == 'CASH_OUT' and row['amount'] > 200000:
        suspicious_flags.append("Large_Cash_Out")
    
    # Rule 4: Account draining pattern
    if row['oldbalanceOrg'] > 0 and row['newbalanceOrig'] == 0 and row['amount'] == row['oldbalanceOrg']:
        suspicious_flags.append("Account_Drained")
    
    # Rule 5: Round amount patterns (potential money laundering)
    if row['amount'] % 50000 == 0 and row['amount'] >= 100000:
        suspicious_flags.append("Round_Large_Amount")
    
    # Rule 6: Destination balance inconsistency
    if pd.notna(row['oldbalanceDest']) and pd.notna(row['newbalanceDest']):
        expected_dest_balance = row['oldbalanceDest'] + row['amount']
        if abs(expected_dest_balance - row['newbalanceDest']) > 0.01:
            suspicious_flags.append("Dest_Balance_Inconsistency")
    
    # Rule 7: Zero balance transactions (suspicious activity on dormant accounts)
    if row['oldbalanceOrg'] == 0 and row['amount'] > 0:
        suspicious_flags.append("Zero_Balance_Transaction")
    
    if len(suspicious_flags) > 0:
        return "Suspicious", suspicious_flags
    else:
        return "Normal", []

# Apply fraud detection rules
print("\n" + "="*50)
print("APPLYING EXPLICIT FRAUD DETECTION RULES")
print("="*50)

fraud_results = data.apply(detect_fraud_rules, axis=1)
data['rule_based_prediction'] = fraud_results.apply(lambda x: x[0])
data['suspicious_flags'] = fraud_results.apply(lambda x: x[1])

# Analysis of rule-based detection
print(f"\nRule-based detection results:")
print(data['rule_based_prediction'].value_counts())

# Check how well rules align with actual fraud labels
rule_vs_actual = pd.crosstab(data['rule_based_prediction'], data['isFraud'], margins=True)
print(f"\nRule-based vs Actual Fraud:")
print(rule_vs_actual)

# Show examples of suspicious transactions caught by rules
suspicious_transactions = data[data['rule_based_prediction'] == 'Suspicious'].head()
print(f"\nExamples of rule-flagged suspicious transactions:")
for idx, row in suspicious_transactions.iterrows():
    print(f"Transaction {idx}: Flags = {row['suspicious_flags']}")
    print(f"  Type: {row['type']}, Amount: ${row['amount']:,.2f}")
    print(f"  Old Balance: ${row['oldbalanceOrg']:,.2f} -> New Balance: ${row['newbalanceOrig']:,.2f}")
    print(f"  Actual Fraud Label: {row['isFraud']}")
    print()

# =============================================================================
# FEATURE ENGINEERING - EXPANDED FEATURE SET
# =============================================================================

print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Convert categorical data to numeric BEFORE correlation analysis
data["type_encoded"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                        "CASH_IN": 3, "TRANSFER": 4,
                                        "DEBIT": 5})

# Create new engineered features
print("Creating engineered features...")

# 1. Balance-related features
data['balance_change'] = data['oldbalanceOrg'] - data['newbalanceOrig']
data['balance_change_matches_amount'] = (abs(data['balance_change'] - data['amount']) < 0.01).astype(int)
data['account_drained'] = ((data['newbalanceOrig'] == 0) & (data['oldbalanceOrg'] > 0)).astype(int)
data['transaction_to_balance_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1)

# 2. Amount-related features
data['log_amount'] = np.log(data['amount'] + 1)
data['is_round_amount'] = (data['amount'] % 1000 == 0).astype(int)
data['is_large_amount'] = (data['amount'] > 100000).astype(int)

# 3. Time-based features (using step)
data['hour_of_day'] = data['step'] % 24
data['day_of_week'] = (data['step'] // 24) % 7
data['is_weekend'] = (data['day_of_week'].isin([5, 6])).astype(int)
data['is_night_time'] = ((data['hour_of_day'] >= 22) | (data['hour_of_day'] <= 6)).astype(int)

# 4. Destination-related features
dest_fraud_rate = data.groupby('nameDest')['isFraud'].mean()
data['dest_fraud_rate'] = data['nameDest'].map(dest_fraud_rate).fillna(0)

# 5. Impossible transaction flags
data['impossible_transaction'] = ((data['amount'] > data['oldbalanceOrg']) & 
                                 (data['oldbalanceOrg'] > 0) & 
                                 (data['newbalanceOrig'] >= 0)).astype(int)

# 6. Destination balance featur
# 
# es
data['dest_balance_change'] = data['newbalanceDest'] - data['oldbalanceDest']
data['dest_balance_consistent'] = (abs(data['dest_balance_change'] - data['amount']) < 0.01).astype(int)

print(f"Original features: 10")
print(f"Engineered features: {len([col for col in data.columns if col not in ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']])}")

# Show correlation analysis
correlation = data.select_dtypes(include=[np.number]).corr()
print(f"\nTop features correlated with fraud:")
fraud_correlations = correlation["isFraud"].abs().sort_values(ascending=False)
print(fraud_correlations.head(10))

# =============================================================================
# MACHINE LEARNING WITH EXPANDED FEATURES
# =============================================================================

print("\n" + "="*50)
print("MACHINE LEARNING MODEL TRAINING")
print("="*50)

# Prepare expanded feature set
feature_columns = [
    'type_encoded', 'amount', 'log_amount', 'oldbalanceOrg', 'newbalanceOrig',
    'step', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_night_time',
    'balance_change', 'balance_change_matches_amount', 'account_drained',
    'transaction_to_balance_ratio', 'is_round_amount', 'is_large_amount',
    'dest_fraud_rate', 'impossible_transaction', 'oldbalanceDest', 'newbalanceDest',
    'dest_balance_change', 'dest_balance_consistent'
]

# Handle missing values
for col in feature_columns:
    if col in data.columns:
        data[col] = data[col].fillna(0)

# Prepare features and target
X = data[feature_columns].fillna(0)
y = data['isFraud']

print(f"Using {len(feature_columns)} features:")
print(feature_columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

# Train the model
model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Fraud', 'Fraud']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# =============================================================================
# TESTING WITH EXPLICIT EXAMPLES
# =============================================================================

print("\n" + "="*50)
print("TESTING WITH EXAMPLE TRANSACTIONS")
print("="*50)

# Test case 1: Your original example
test_case_1 = np.array([[4, 9000.60, 9000.60, 0.0, 100, 4, 0, 0, 0, 
                        9000.60, 1, 1, 1.0, 0, 0, 0.0, 1, 0, 0, 0, 0]])

# Test case 2: Normal transaction
test_case_2 = np.array([[2, 100.00, 5000.00, 4900.00, 50, 14, 2, 0, 0,
                        -100.00, 1, 0, 0.02, 0, 0, 0.1, 0, 1000, 1100, 100, 1]])

# Test case 3: Suspicious large round amount
test_case_3 = np.array([[1, 500000.00, 600000.00, 100000.00, 200, 2, 3, 0, 0,
                        -500000.00, 1, 0, 0.83, 1, 1, 0.8, 0, 0, 500000, 500000, 1]])

test_cases = [
    ("Account Draining (Your Example)", test_case_1),
    ("Normal Small Transaction", test_case_2),
    ("Large Suspicious Cash Out", test_case_3)
]

for name, features in test_cases:
    # Ensure we have the right number of features
    if features.shape[1] == len(feature_columns):
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        print(f"\n{name}:")
        print(f"  Prediction: {'FRAUD' if prediction == 1 else 'NO FRAUD'}")
        print(f"  Fraud Probability: {probability[1]:.4f}")
        print(f"  Features: {features[0][:6]}...")  # Show first 6 features
    else:
        print(f"\n{name}: Feature dimension mismatch")

print(f"\nModel trained on {len(X_train)} transactions")
print(f"Tested on {len(X_test)} transactions")
print(f"Total fraud cases in training: {sum(y_train)}")
print(f"Total fraud cases in testing: {sum(y_test)}")