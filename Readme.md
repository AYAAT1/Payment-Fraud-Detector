# Financial Transaction Fraud Detection System

A comprehensive fraud detection system using Decision Trees with rule-based detection and advanced feature engineering for financial transaction analysis.

## 🎯 Project Overview

This project implements a robust fraud detection system that combines:
- **Rule-based fraud detection** using business logic
- **Machine learning** with Decision Trees
- **Advanced feature engineering** for better detection accuracy
- **Explainable AI** for regulatory compliance

## 📊 Dataset

The project uses the **Online Payment Fraud Detection Dataset** available on Kaggle :

**📥 Download Link**: [Online Payment Fraud Detection Dataset](https://www.kaggle.com/code/seuwenfei/online-payment-fraud-detection/input?scriptVersionId=127055269)

**Dataset Details:**
- **Source**: Kaggle - Online Payment Fraud Detection
- **File**: `payment_dataset.csv`
- **Size**: Contains financial transaction records with fraud labels
- **Features**: Transaction types, amounts, account balances, timestamps, and fraud indicators
-  **Type**: This is a **time series dataset**, where each transaction is recorded in time order using the `step` feature, which represents the hour of the transaction since the start of the dataset.
- **Use Case**: Real-world financial transaction data for fraud detection research

**Note**: Download the dataset from Kaggle and place `payment_dataset.csv` in your project directory before running the code.

## 🤔 Why Decision Trees for Fraud Detection?

### 1. **Natural Rule-Based Logic**
Decision trees mirror how fraud analysts think:
```
IF transaction_type == "CASH_OUT" 
   AND amount > 200000 
   AND account_balance_after == 0
THEN high_fraud_probability
```

### 2. **Highly Interpretable and Explainable**
Critical for financial compliance - you can explain every decision:
```
Why was this transaction flagged as fraud?
└── Amount > $50,000 (Yes)
    └── Transaction Type = CASH_OUT (Yes)
        └── New Balance = $0 (Yes)
            └── FRAUD (95% confidence)
```

### 3. **Handles Mixed Data Types Naturally**
- **No preprocessing needed** for categorical variables
- **No scaling required** for numerical features
- **Automatically handles** different data types in the same model

### 4. **Captures Non-Linear Patterns**
```python
# Decision tree captures complex patterns:
IF account_age < 30_days:
    IF amount < 100: FRAUD
    IF amount > 100: LEGITIMATE
```

### 5. **Additional Benefits**
- **Fast prediction times** (O(log n)) for real-time detection
- **Robust to outliers** in financial data
- **Automatic feature selection**
- **Handles imbalanced data** well
- **No assumptions** about data distribution

## 📁 Project Structure

```
fraud-detection/
│
├── payment_dataset.csv           # Input dataset
├── fraud_detection.py           # Main implementation
├── README.md                    # This file
│
└── Code Structure:
    ├── Data Loading & Exploration
    ├── Rule-Based Fraud Detection
    ├── Feature Engineering
    ├── Machine Learning Model
    └── Testing & Evaluation
```

## 🔧 Code Structure Breakdown

### 1. **Data Loading & Exploration**
```python
# Load and explore the dataset
data = pd.read_csv('payment_dataset.csv')
print(f"Dataset shape: {data.shape}")
print(data.head())
```

### 2. **Rule-Based Fraud Detection**
Implements 7 explicit business rules:
```python
def detect_fraud_rules(row):
    """Apply explicit business rules to detect suspicious transactions"""
    suspicious_flags = []
    
    # Rule 1: Balance inconsistency
    # Rule 2: Impossible transaction  
    # Rule 3: Large cash out operations
    # Rule 4: Account draining pattern
    # Rule 5: Round amount patterns
    # Rule 6: Destination balance inconsistency
    # Rule 7: Zero balance transactions
```

**Fraud Detection Rules:**
- **Balance Inconsistency**: Mathematical impossibility in balance calculations
- **Impossible Transactions**: Spending more than available balance
- **Large Cash Outs**: Cash withdrawals exceeding $200,000
- **Account Draining**: Complete balance depletion in single transaction
- **Round Large Amounts**: Potential money laundering patterns
- **Destination Balance Issues**: Inconsistent recipient balance changes
- **Zero Balance Activity**: Suspicious activity on dormant accounts

### 3. **Feature Engineering**
Creates 22+ engineered features across categories:

**Balance-Related Features:**
```python
data['balance_change'] = data['oldbalanceOrg'] - data['newbalanceOrig']
data['account_drained'] = ((data['newbalanceOrig'] == 0) & (data['oldbalanceOrg'] > 0))
data['transaction_to_balance_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1)
```

**Amount-Related Features:**
```python
data['log_amount'] = np.log(data['amount'] + 1)
data['is_round_amount'] = (data['amount'] % 1000 == 0)
data['is_large_amount'] = (data['amount'] > 100000)
```

**Time-Based Features:**
```python
data['hour_of_day'] = data['step'] % 24
data['day_of_week'] = (data['step'] // 24) % 7
data['is_weekend'] = (data['day_of_week'].isin([5, 6]))
```

**Risk-Based Features:**
```python
dest_fraud_rate = data.groupby('nameDest')['isFraud'].mean()
data['dest_fraud_rate'] = data['nameDest'].map(dest_fraud_rate)
```

### 4. **Machine Learning Model**
```python
# Configure Decision Tree with optimal parameters
model = DecisionTreeClassifier(
    random_state=42, 
    max_depth=10,           # Prevent overfitting
    min_samples_split=100   # Ensure statistical significance
)

# Train on engineered features
X = data[feature_columns].fillna(0)
y = data['isFraud']
model.fit(X_train, y_train)
```

### 5. **Model Evaluation & Testing**
```python
# Performance metrics
accuracy = model.score(X_test, y_test)
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Test with specific examples
test_cases = [
    ("Account Draining Pattern", account_drain_features),
    ("Normal Transaction", normal_features),
    ("Large Suspicious Cash Out", suspicious_features)
]
```

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Dataset Setup
1. **Download the dataset** from Kaggle:
   - Visit: https://www.kaggle.com/code/seuwenfei/online-payment-fraud-detection/input?scriptVersionId=127055269
   - Download `payment_dataset.csv`
   - Place the file in your project directory

2. **Alternative**: If you have Kaggle API configured:
```bash
kaggle datasets download -d seuwenfei/online-payment-fraud-detection
```

### Running the Code
```python
python fraud_detection.py
```

### Expected Output
1. **Dataset Analysis**: Shape, transaction types, missing values
2. **Rule-Based Results**: Suspicious transactions flagged by business rules
3. **Feature Engineering**: Correlation analysis and feature importance
4. **Model Performance**: Accuracy, precision, recall, F1-score
5. **Test Cases**: Predictions on example transactions

## 📊 Key Features

### Input Features (22 total):
- **Original**: `type_encoded`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `step`
- **Engineered**: `log_amount`, `account_drained`, `transaction_to_balance_ratio`
- **Time-based**: `hour_of_day`, `day_of_week`, `is_weekend`, `is_night_time`
- **Risk-based**: `dest_fraud_rate`, `impossible_transaction`
- **Pattern-based**: `is_round_amount`, `is_large_amount`

### Model Configuration:
- **Algorithm**: Decision Tree Classifier
- **Max Depth**: 10 (prevents overfitting)
- **Min Samples Split**: 100 (ensures statistical significance)
- **Train/Test Split**: 90/10 with stratification

## 🎯 Test Cases

The system includes three test scenarios:

1. **Account Draining Pattern** (High Risk)
   - Transaction equals full account balance
   - Results in zero final balance
   - Transfer type transaction

2. **Normal Small Transaction** (Low Risk)
   - Regular payment amount
   - Reasonable balance changes
   - Consistent with spending patterns

3. **Large Suspicious Cash Out** (High Risk)
   - Large round amount withdrawal
   - High transaction-to-balance ratio
   - Cash-out transaction type

## 📈 Performance Metrics

The model provides:
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score for both classes
- **Feature Importance**: Which features contribute most to fraud detection
- **Confusion Matrix**: True/false positive and negative breakdown

## 🔍 Model Interpretability

Every prediction can be explained through the decision tree path:
```
Transaction Analysis:
├── Is amount > $50,000? → Yes
├── Is transaction type CASH_OUT? → Yes  
├── Does new balance = $0? → Yes
└── PREDICTION: FRAUD (Confidence: 95%)
```

## 🚨 Business Rules Integration

The system combines ML predictions with explicit business rules:
- Mathematical consistency checks
- Regulatory compliance patterns
- Domain expertise encoded as rules
- Dual-layer fraud detection approach

## ⚡ Production Considerations

- **Real-time Processing**: O(log n) prediction time
- **Explainable Decisions**: Full audit trail for each prediction
- **Regulatory Compliance**: Transparent decision-making process
- **Scalable Architecture**: Handles high-volume transaction streams
- **Regular Retraining**: Adapts to evolving fraud patterns

## 📝 Next Steps

1. **Ensemble Methods**: Combine with Random Forest for higher accuracy
2. **Real-time Integration**: Deploy for live transaction monitoring
3. **Advanced Features**: Add network analysis and behavioral patterns
4. **Continuous Learning**: Implement feedback loops for model improvement
5. **API Development**: Create REST API for production deployment

## 🤝 Contributing

Feel free to contribute by:
- Adding new fraud detection rules
- Improving feature engineering
- Optimizing model parameters
- Enhancing documentation

---

**Note**: This system is designed for educational and research purposes. Production deployment requires additional security, privacy, and compliance considerations.