# 📘 AV Demand Forecasting - Inference Guide & Technical Documentation

**Version:** 1.0  
**Last Updated:** November 2025
**Model Type:** LightGBM Regressor with 10-Fold Cross Validation

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Overview](#model-overview)
3. [Data Structure & Requirements](#data-structure--requirements)
4. [Feature Engineering Pipeline](#feature-engineering-pipeline)
5. [Inference Process](#inference-process)
6. [Input Specifications](#input-specifications)
7. [Output Specifications](#output-specifications)
8. [Code Examples](#code-examples)
9. [Performance Metrics](#performance-metrics)
10. [Troubleshooting](#troubleshooting)
11. [Appendix](#appendix)

---

## 1. Executive Summary

### 1.1 Purpose
Model này dự đoán **units_sold** (số lượng sản phẩm bán được) cho mỗi combination của (store_id, sku_id, week).

### 1.2 Key Metrics
- **Validation RMSLE:** ~327
- **Expected Public LB:** ~360-365
- **Training Time:** ~15-20 phút (10-fold CV)
- **Inference Time:** <1 giây cho 13,860 samples

### 1.3 Technology Stack
```
Python 3.7+
├── pandas >= 1.3.0
├── numpy >= 1.21.0
├── lightgbm >= 3.3.0
├── category-encoders >= 2.5.0
└── scikit-learn >= 1.0.0
```

---

## 2. Model Overview

### 2.1 Model Architecture

```
Input Data (Test Set)
    ↓
Feature Engineering Pipeline
    ├── Price Features (3 features)
    ├── Categorical Encoding (4 features)
    └── DateTime Features (12 features)
    ↓
Total: 23 Features
    ↓
LightGBM DART Regressor
    ├── 10-Fold Cross Validation
    ├── 575 iterations per fold
    └── Early stopping (30 rounds)
    ↓
Ensemble Predictions (average of 10 models)
    ↓
Output: units_sold predictions
```

### 2.2 Model Hyperparameters

```python
params = {
    'boosting_type': 'dart',           # Dropout for trees
    'objective': 'regression',         # Regression task
    'metric': 'l1',                    # MAE metric
    'learning_rate': 0.5,              # High learning rate
    'min_data_in_leaf': 15,            # Minimum samples per leaf
    'bagging_fraction': 0.7,           # 70% data sampling
    'feature_fraction': 0.7,           # 70% feature sampling
    'bagging_seed': 50                 # Random seed
}
```

### 2.3 Target Transformation

**Important:** Target variable được transform bằng **log1p** (log(x+1)) trước khi training.

```python
# Training
y_train_transformed = np.log1p(y_train)

# Prediction (phải inverse transform)
y_pred_original = np.exp(y_pred_transformed)
```

**Lý do:**
- Target distribution bị skewed (phân phối lệch)
- Log transformation làm phân phối gần Normal hơn
- Giúp model học tốt hơn

---

## 3. Data Structure & Requirements

### 3.1 Input Data Format

**File:** CSV format với các columns bắt buộc

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `record_ID` | int64 | Unique identifier cho mỗi record | 212645 |
| `week` | string | Ngày bắt đầu tuần (format: DD/MM/YY) | "16/07/13" |
| `store_id` | int64 | ID của cửa hàng | 8091 |
| `sku_id` | int64 | ID của sản phẩm (SKU) | 216418 |
| `total_price` | float64 | Giá bán thực tế | 108.30 |
| `base_price` | float64 | Giá gốc | 108.30 |
| `is_featured_sku` | int64 | Sản phẩm có được feature không (0/1) | 0 |
| `is_display_sku` | int64 | Sản phẩm có được display không (0/1) | 0 |

**Lưu ý:**
- `week` phải theo format `DD/MM/YY` (ví dụ: 16/07/13)
- `is_featured_sku` và `is_display_sku` chỉ nhận giá trị 0 hoặc 1
- Không được có missing values trong các columns trừ `total_price`

### 3.2 Sample Input Data

```csv
record_ID,week,store_id,sku_id,total_price,base_price,is_featured_sku,is_display_sku
212645,16/07/13,8091,216418,108.3000,108.3000,0,0
212646,16/07/13,8091,216419,109.0125,109.0125,0,0
212647,16/07/13,8091,216425,133.9500,133.9500,0,0
```

---

## 4. Feature Engineering Pipeline

### 4.1 Overview

Từ 8 features gốc, model tạo ra **23 features** thông qua feature engineering:

```
Original Features (8)
    ├── record_ID (dropped)
    ├── week (transformed to 6 datetime features)
    ├── store_id (kept + encoded)
    ├── sku_id (kept + encoded)
    ├── total_price (kept + derived)
    ├── base_price (kept + derived)
    ├── is_featured_sku (kept)
    └── is_display_sku (kept)
        ↓
Engineered Features (23)
```

### 4.2 Price Features (3 features)

#### 4.2.1 `diff`
**Định nghĩa:** Chênh lệch giữa base_price và total_price

```python
diff = base_price - total_price
```

**Ý nghĩa:**
- Mức discount (giảm giá)
- `diff > 0`: Có discount
- `diff = 0`: Không có discount
- `diff < 0`: Tăng giá (hiếm)

**Example:**
```
base_price = 111.86, total_price = 99.04
diff = 111.86 - 99.04 = 12.82
→ Sản phẩm được giảm 12.82 đơn vị tiền
```

#### 4.2.2 `relative_diff_base`
**Định nghĩa:** % discount so với base_price

```python
relative_diff_base = diff / base_price
```

**Ý nghĩa:**
- Tỷ lệ discount
- Range: [0, 1] thường
- Cao = discount nhiều

**Example:**
```
diff = 12.82, base_price = 111.86
relative_diff_base = 12.82 / 111.86 = 0.1147 (11.47% discount)
```

#### 4.2.3 `relative_diff_total`
**Định nghĩa:** Markup ratio so với total_price

```python
relative_diff_total = diff / total_price
```

**Ý nghĩa:**
- Tỷ lệ discount tính theo giá bán
- Thường cao hơn `relative_diff_base`

**Example:**
```
diff = 12.82, total_price = 99.04
relative_diff_total = 12.82 / 99.04 = 0.1295 (12.95%)
```

### 4.3 Categorical Encoding (4 features)

#### 4.3.1 M-Estimate Encoding

**Technique:** Target-based encoding using M-Estimate

**Formula:**
```
encoded_value = (n * mean_target + m * global_mean) / (n + m)

where:
- n = số lượng samples có value đó
- mean_target = mean của target cho value đó
- global_mean = mean toàn bộ dataset
- m = regularization parameter (default=1)
```

**Features:**
1. `store_encoded`: M-estimate encoding của store_id
2. `sku_encoded`: M-estimate encoding của sku_id
3. `store_id`: Original ID (kept as categorical)
4. `sku_id`: Original ID (kept as categorical)

**Why M-Estimate?**
- Handle high cardinality (76 stores, 28 SKUs)
- Prevent overfitting
- Better than one-hot encoding
- Incorporate target information

**Example:**
```python
# Store 8091 xuất hiện 130 lần trong training với mean units_sold = 52.3
# Global mean units_sold = 51.67

store_encoded = (130 * 52.3 + 1 * 51.67) / (130 + 1)
              = 6850.67 / 131
              = 52.29
```

### 4.4 DateTime Features (12 features)

**Base Date:** 17/01/2011 (reference point)

#### 4.4.1 Week Start Features (6 features)

| Feature | Description | Range | Example |
|---------|-------------|-------|---------|
| `year` | Năm của week | 2011-2013 | 2013 |
| `month` | Tháng của week | 1-12 | 7 |
| `date` | Ngày trong tháng | 1-31 | 16 |
| `weekday` | Thứ trong tuần (0=Monday) | 0-6 | 1 (Tuesday) |
| `weeknum` | Tuần thứ mấy trong năm | 1-53 | 28 |
| `week_serial` | Số tuần kể từ base date | 0-140 | 129.14 |

**Calculation Example:**
```python
week = "16/07/13" → datetime(2013, 7, 16)
base_date = datetime(2011, 1, 17)

week_serial = (week - base_date).days / 7
            = 911 / 7
            = 129.14 weeks
```

#### 4.4.2 Week End Features (6 features)

**Concept:** Weekend date = Week start + 6 days

| Feature | Description | Range | Example |
|---------|-------------|-------|---------|
| `end_year` | Năm của weekend | 2011-2013 | 2013 |
| `end_month` | Tháng của weekend | 1-12 | 7 |
| `end_date` | Ngày của weekend | 1-31 | 22 |
| `end_weekday` | Thứ của weekend (0=Monday) | 0-6 | 0 (Monday) |
| `end_weeknum` | Tuần của weekend | 1-53 | 29 |
| `end_week_serial` | Serial number | 0-141 | 130.0 |

**Why Week End Features?**
- Capture weekly patterns
- Some stores có behavior khác ngày cuối tuần
- Seasonality detection

### 4.5 Feature Summary Table

| Category | Count | Features |
|----------|-------|----------|
| **Original** | 5 | base_price, total_price, is_featured_sku, is_display_sku, store_id, sku_id |
| **Price Derived** | 3 | diff, relative_diff_base, relative_diff_total |
| **Categorical Encoded** | 2 | store_encoded, sku_encoded |
| **DateTime** | 12 | year, month, date, weekday, weeknum, week_serial, end_year, end_month, end_date, end_weekday, end_weeknum, end_week_serial |
| **TOTAL** | 23 | - |

---

## 5. Inference Process

### 5.1 Complete Pipeline Flowchart

```
┌─────────────────────────────────────────┐
│  1. Load Raw Test Data (CSV)           │
│     - 8 columns                         │
│     - 13,860 rows                       │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  2. Data Validation                     │
│     - Check required columns            │
│     - Check data types                  │
│     - Check value ranges                │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  3. Data Preprocessing                  │
│     - Fill missing total_price          │
│     - Create store_sku identifier       │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  4. Feature Engineering                 │
│     ├─ Price Features                   │
│     ├─ Categorical Encoding             │
│     └─ DateTime Features                │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  5. Load Trained Models                 │
│     - 10 LightGBM models (from CV)      │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  6. Make Predictions                    │
│     - Predict với mỗi fold model        │
│     - Transform: np.exp(predictions)    │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  7. Ensemble Predictions                │
│     - Average 10 predictions            │
│     - Apply np.abs() (ensure positive)  │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  8. Format Output                       │
│     - record_ID, units_sold             │
│     - Save to CSV                       │
└─────────────────────────────────────────┘
```

### 5.2 Step-by-Step Inference Guide

#### Step 1: Load Test Data

```python
import pandas as pd

# Load test data
test = pd.read_csv('test_data.csv')

# Verify shape
print(f"Test data shape: {test.shape}")
# Expected: (n_samples, 8)
```

#### Step 2: Data Validation

```python
# Required columns
required_cols = ['record_ID', 'week', 'store_id', 'sku_id', 
                 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku']

# Check columns
assert all(col in test.columns for col in required_cols), "Missing required columns!"

# Check data types
assert test['store_id'].dtype == 'int64', "store_id must be int64"
assert test['sku_id'].dtype == 'int64', "sku_id must be int64"

# Check value ranges
assert test['is_featured_sku'].isin([0, 1]).all(), "is_featured_sku must be 0 or 1"
assert test['is_display_sku'].isin([0, 1]).all(), "is_display_sku must be 0 or 1"

print("✓ Data validation passed!")
```

#### Step 3: Data Preprocessing

```python
# Fill missing total_price with base_price
test['total_price'] = test['total_price'].fillna(test['base_price'])

# Create store_sku identifier (optional, for tracking)
test['store_sku'] = (test['store_id'].astype('str') + "_" + 
                     test['sku_id'].astype('str'))
```

#### Step 4: Feature Engineering

**A. Price Features**
```python
test['diff'] = test['base_price'] - test['total_price']
test['relative_diff_base'] = test['diff'] / test['base_price']
test['relative_diff_total'] = test['diff'] / test['total_price']
```

**B. Categorical Encoding**
```python
from category_encoders import MEstimateEncoder

# IMPORTANT: Must use pre-fitted encoders from training!
# Don't fit on test data!

# Load pre-fitted encoders
import joblib
store_encoder = joblib.load('store_encoder.pkl')
sku_encoder = joblib.load('sku_encoder.pkl')

# Transform
test['store_encoded'] = store_encoder.transform(test['store_id'])
test['sku_encoded'] = sku_encoder.transform(test['sku_id'])
```

**C. DateTime Features**
```python
from datetime import datetime, timedelta

# Convert to datetime
test['week'] = pd.to_datetime(test['week'], format='%d/%m/%y')
test['weekend_date'] = test['week'] + timedelta(days=6)

# Extract features
start_date = datetime(2011, 1, 17)

test['year'] = test['week'].dt.year
test['date'] = test['week'].dt.day
test['month'] = test['week'].dt.month
test['weekday'] = test['week'].dt.dayofweek
test['weeknum'] = test['week'].dt.isocalendar().week
test['week_serial'] = (test['week'] - start_date).dt.total_seconds() / (86400 * 7)

test['end_year'] = test['weekend_date'].dt.year
test['end_date'] = test['weekend_date'].dt.day
test['end_month'] = test['weekend_date'].dt.month
test['end_weekday'] = test['weekend_date'].dt.dayofweek
test['end_weeknum'] = test['weekend_date'].dt.isocalendar().week
test['end_week_serial'] = (test['weekend_date'] - start_date).dt.total_seconds() / (86400 * 7)

# Apply M-Estimate encoding to time features (use pre-fitted encoder)
time_encoder = joblib.load('time_encoder.pkl')
time_features = ['date', 'end_week_serial', 'month', 'week_serial', 'year', 
                'weekday', 'weeknum', 'end_weekday', 'end_month', 
                'end_weeknum', 'end_date', 'end_year']
test[time_features] = time_encoder.transform(test[time_features])
```

#### Step 5: Prepare Features

```python
# Feature columns used by model
cols_to_use = [
    'base_price', 'total_price', 'diff', 'relative_diff_base', 'relative_diff_total',
    'is_featured_sku', 'is_display_sku', 'store_encoded', 'sku_encoded',
    'store_id', 'sku_id',
    'date', 'end_week_serial', 'month', 'week_serial', 'year', 'weekday', 
    'weeknum', 'end_weekday', 'end_month', 'end_weeknum', 'end_date', 'end_year'
]

X_test = test[cols_to_use]
```

#### Step 6: Load Models & Predict

```python
import lightgbm as lgb
import numpy as np

# Load all 10 fold models
models = []
for fold in range(10):
    model = lgb.Booster(model_file=f'model_fold_{fold}.txt')
    models.append(model)

# Make predictions with each model
predictions_all = []
for model in models:
    pred = model.predict(X_test, num_iteration=model.best_iteration)
    # Inverse log transform
    pred = np.exp(pred)
    predictions_all.append(pred)

# Ensemble: average predictions
final_predictions = np.mean(predictions_all, axis=0)

# Ensure positive values
final_predictions = np.abs(final_predictions)
```

#### Step 7: Format Output

```python
# Create submission
submission = pd.DataFrame({
    'record_ID': test['record_ID'],
    'units_sold': final_predictions
})

# Save
submission.to_csv('submission.csv', index=False)
print(f"✓ Saved {len(submission)} predictions to submission.csv")
```

### 5.3 Complete Inference Script

```python
def inference_pipeline(test_path, output_path='submission.csv'):
    """
    Complete inference pipeline
    
    Args:
        test_path: Path to test CSV file
        output_path: Path to save predictions
        
    Returns:
        DataFrame with predictions
    """
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from datetime import datetime, timedelta
    import joblib
    
    # 1. Load data
    test = pd.read_csv(test_path)
    print(f"Loaded {len(test)} test samples")
    
    # 2. Preprocessing
    test['total_price'] = test['total_price'].fillna(test['base_price'])
    
    # 3. Feature Engineering
    # Price features
    test['diff'] = test['base_price'] - test['total_price']
    test['relative_diff_base'] = test['diff'] / test['base_price']
    test['relative_diff_total'] = test['diff'] / test['total_price']
    
    # Categorical encoding
    store_encoder = joblib.load('store_encoder.pkl')
    sku_encoder = joblib.load('sku_encoder.pkl')
    test['store_encoded'] = store_encoder.transform(test['store_id'])
    test['sku_encoded'] = sku_encoder.transform(test['sku_id'])
    
    # DateTime features
    test['week'] = pd.to_datetime(test['week'], format='%d/%m/%y')
    test['weekend_date'] = test['week'] + timedelta(days=6)
    start_date = datetime(2011, 1, 17)
    
    test['year'] = test['week'].dt.year
    test['date'] = test['week'].dt.day
    test['month'] = test['week'].dt.month
    test['weekday'] = test['week'].dt.dayofweek
    test['weeknum'] = test['week'].dt.isocalendar().week
    test['week_serial'] = (test['week'] - start_date).dt.total_seconds() / (86400 * 7)
    
    test['end_year'] = test['weekend_date'].dt.year
    test['end_date'] = test['weekend_date'].dt.day
    test['end_month'] = test['weekend_date'].dt.month
    test['end_weekday'] = test['weekend_date'].dt.dayofweek
    test['end_weeknum'] = test['weekend_date'].dt.isocalendar().week
    test['end_week_serial'] = (test['weekend_date'] - start_date).dt.total_seconds() / (86400 * 7)
    
    time_encoder = joblib.load('time_encoder.pkl')
    time_features = ['date', 'end_week_serial', 'month', 'week_serial', 'year', 
                    'weekday', 'weeknum', 'end_weekday', 'end_month', 
                    'end_weeknum', 'end_date', 'end_year']
    test[time_features] = time_encoder.transform(test[time_features])
    
    # 4. Prepare features
    cols_to_use = [
        'base_price', 'total_price', 'diff', 'relative_diff_base', 'relative_diff_total',
        'is_featured_sku', 'is_display_sku', 'store_encoded', 'sku_encoded',
        'store_id', 'sku_id',
        'date', 'end_week_serial', 'month', 'week_serial', 'year', 'weekday', 
        'weeknum', 'end_weekday', 'end_month', 'end_weeknum', 'end_date', 'end_year'
    ]
    X_test = test[cols_to_use]
    
    # 5. Load models and predict
    predictions_all = []
    for fold in range(10):
        model = lgb.Booster(model_file=f'model_fold_{fold}.txt')
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        pred = np.exp(pred)  # Inverse transform
        predictions_all.append(pred)
    
    # 6. Ensemble
    final_predictions = np.mean(predictions_all, axis=0)
    final_predictions = np.abs(final_predictions)
    
    # 7. Format output
    submission = pd.DataFrame({
        'record_ID': test['record_ID'],
        'units_sold': final_predictions
    })
    
    # 8. Save
    submission.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to {output_path}")
    
    return submission

# Usage
predictions = inference_pipeline('test.csv', 'submission.csv')
```

---

## 6. Input Specifications

### 6.1 File Format

**Format:** CSV (Comma Separated Values)  
**Encoding:** UTF-8  
**Line Ending:** LF (\n) hoặc CRLF (\r\n)  
**Header:** Required (first row)

### 6.2 Column Specifications

| Column | Type | Nullable | Min | Max | Format | Notes |
|--------|------|----------|-----|-----|--------|-------|
| record_ID | int64 | No | 1 | 999999 | Integer | Unique identifier |
| week | object/string | No | - | - | DD/MM/YY | Ngày bắt đầu tuần |
| store_id | int64 | No | 8023 | 9984 | Integer | 76 unique stores |
| sku_id | int64 | No | 216233 | 679023 | Integer | 28 unique SKUs |
| total_price | float64 | Yes | 41.33 | 562.16 | Float | Giá bán |
| base_price | float64 | No | 61.28 | 562.16 | Float | Giá gốc |
| is_featured_sku | int64 | No | 0 | 1 | Binary | Feature flag |
| is_display_sku | int64 | No | 0 | 1 | Binary | Display flag |

### 6.3 Data Constraints

**Business Rules:**
1. `base_price >= total_price` (usually, có thể có exceptions)
2. `total_price > 0` và `base_price > 0`
3. Mỗi (store_id, sku_id, week) combination là unique
4. `week` phải theo format DD/MM/YY (ví dụ: 16/07/13)

**Technical Constraints:**
1. Maximum file size: 100 MB
2. Maximum rows: 1,000,000
3. Character encoding: UTF-8
4. No special characters trong column names

### 6.4 Missing Value Handling

| Column | Missing Value Strategy |
|--------|----------------------|
| total_price | Fill với base_price |
| Others | **NOT ALLOWED** - sẽ raise error |

### 6.5 Example Valid Input

```csv
record_ID,week,store_id,sku_id,total_price,base_price,is_featured_sku,is_display_sku
212645,16/07/13,8091,216418,108.30,108.30,0,0
212646,16/07/13,8091,216419,109.01,109.01,0,0
212647,16/07/13,8091,216425,120.50,133.95,1,1
212648,16/07/13,8091,216233,,133.95,0,0
```

**Note:** Row 212648 có missing total_price → sẽ được fill = base_price = 133.95

---

## 7. Output Specifications

### 7.1 File Format

**Format:** CSV  
**Encoding:** UTF-8  
**Columns:** 2 columns (record_ID, units_sold)  
**Rows:** Same as input (13,860 for test set)

### 7.2 Output Schema

| Column | Type | Range | Precision | Description |
|--------|------|-------|-----------|-------------|
| record_ID | int64 | - | - | Copy từ input, unique identifier |
| units_sold | float64 | [0, ∞) | 6 decimals | Predicted số lượng bán |

### 7.3 Output Statistics (Expected)

```
units_sold:
    count: 13,860
    mean:  ~45-50
    std:   ~35-40
    min:   ~5-10
    25%:   ~20-25
    50%:   ~35-40
    75%:   ~55-60
    max:   ~200-300
```

### 7.4 Sample Output

```csv
record_ID,units_sold
212645,21.163058
212646,23.675132
212647,34.839319
212648,31.773779
212649,24.315106
```

### 7.5 Post-Processing Rules

1. **Inverse Log Transform:** `units_sold = exp(prediction) - 1`
2. **Absolute Value:** `units_sold = abs(units_sold)` (ensure non-negative)
3. **No Rounding:** Keep as float64 với 6 decimals
4. **No Clipping:** Không clip min/max values

---

## 8. Code Examples

### 8.1 Quick Inference (Minimal Code)

```python
# Assuming trained model saved
import pandas as pd
import lightgbm as lgb
import numpy as np

# Load
test = pd.read_csv('test.csv')
model = lgb.Booster(model_file='model.txt')

# Simple prediction (assume features already prepared)
X_test = test[feature_columns]
predictions = np.exp(model.predict(X_test))

# Save
pd.DataFrame({
    'record_ID': test['record_ID'],
    'units_sold': predictions
}).to_csv('submission.csv', index=False)
```

### 8.2 Batch Inference (Large Dataset)

```python
def batch_inference(test_path, batch_size=1000):
    """Process large dataset in batches"""
    import pandas as pd
    import lightgbm as lgb
    import numpy as np
    
    # Load model once
    model = lgb.Booster(model_file='model.txt')
    
    # Process in chunks
    results = []
    for chunk in pd.read_csv(test_path, chunksize=batch_size):
        # Preprocess chunk
        chunk = preprocess_features(chunk)
        
        # Predict
        X_chunk = chunk[feature_columns]
        pred = np.exp(model.predict(X_chunk))
        
        # Store
        results.append(pd.DataFrame({
            'record_ID': chunk['record_ID'],
            'units_sold': pred
        }))
    
    # Combine
    final = pd.concat(results, ignore_index=True)
    return final
```

### 8.3 Real-time Inference (Single Sample)

```python
def predict_single_sample(sample_dict):
    """
    Predict for single sample
    
    Args:
        sample_dict: Dictionary with required fields
        
    Returns:
        Predicted units_sold
        
    Example:
        sample = {
            'week': '16/07/13',
            'store_id': 8091,
            'sku_id': 216418,
            'total_price': 108.30,
            'base_price': 108.30,
            'is_featured_sku': 0,
            'is_display_sku': 0
        }
        units = predict_single_sample(sample)
    """
    import pandas as pd
    import lightgbm as lgb
    import numpy as np
    
    # Convert to DataFrame
    df = pd.DataFrame([sample_dict])
    
    # Feature engineering
    df = preprocess_features(df)
    
    # Predict
    X = df[feature_columns]
    model = lgb.Booster(model_file='model.txt')
    prediction = np.exp(model.predict(X))[0]
    
    return float(prediction)
```

### 8.4 API Endpoint (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np

app = FastAPI()

# Load model at startup
model = lgb.Booster(model_file='model.txt')

class PredictionRequest(BaseModel):
    week: str
    store_id: int
    sku_id: int
    total_price: float
    base_price: float
    is_featured_sku: int
    is_display_sku: int

class PredictionResponse(BaseModel):
    record_ID: int
    units_sold: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Feature engineering
        features = engineer_features(request.dict())
        
        # Predict
        prediction = np.exp(model.predict([features]))[0]
        
        return PredictionResponse(
            record_ID=0,  # Generate or pass from request
            units_sold=float(prediction)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 9. Performance Metrics

### 9.1 Evaluation Metric

**Primary Metric:** RMSLE (Root Mean Squared Logarithmic Error)

**Formula:**
```
RMSLE = sqrt(mean((log(predicted + 1) - log(actual + 1))^2)) * 1000
```

**Why RMSLE?**
- Handle skewed distribution
- Penalize under-prediction và over-prediction equally (in log space)
- Less sensitive to outliers
- Business-friendly: focus on relative errors

**Python Implementation:**
```python
def RMSLE(actual, predicted):
    predicted = np.array([np.log(np.abs(x+1.0)) for x in predicted])
    actual = np.array([np.log(np.abs(x+1.0)) for x in actual])
    log_err = actual - predicted
    return 1000 * np.sqrt(np.mean(log_err**2))
```

### 9.2 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Validation RMSLE** | 327.76 | 10-fold CV average |
| **Std Dev (across folds)** | ±3.5 | Consistent performance |
| **Min RMSLE** | 323.02 | Best fold |
| **Max RMSLE** | 330.70 | Worst fold |
| **Public LB** | 360.71 | Test set performance |

### 9.3 Inference Speed

**Hardware:** 
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 8GB
- No GPU required

**Benchmarks:**
- Single sample: <1ms
- 1,000 samples: ~50ms
- 13,860 samples: ~0.7s
- Feature engineering: ~2-3s
- Total pipeline: ~4-5s

### 9.4 Model Size

```
Total size: ~15 MB (10 models)
├── model_fold_0.txt: ~1.5 MB
├── model_fold_1.txt: ~1.5 MB
├── ...
├── model_fold_9.txt: ~1.5 MB
├── store_encoder.pkl: ~5 KB
├── sku_encoder.pkl: ~3 KB
└── time_encoder.pkl: ~2 KB
```

---

## 10. Troubleshooting

### 10.1 Common Errors

#### Error 1: KeyError - Missing Column

**Error Message:**
```
KeyError: 'column_name'
```

**Cause:** Input CSV thiếu required column

**Solution:**
```python
# Verify columns
required = ['record_ID', 'week', 'store_id', 'sku_id', 
            'total_price', 'base_price', 'is_featured_sku', 'is_display_sku']
missing = set(required) - set(df.columns)
if missing:
    print(f"Missing columns: {missing}")
```

#### Error 2: DateParseError

**Error Message:**
```
ValueError: time data '2013-07-16' does not match format '%d/%m/%y'
```

**Cause:** Week column có wrong format

**Solution:**
```python
# Check and convert format
df['week'] = pd.to_datetime(df['week'], format='%d/%m/%y')
# or auto-detect
df['week'] = pd.to_datetime(df['week'], infer_datetime_format=True)
```

#### Error 3: Categorical Value Not Seen in Training

**Error Message:**
```
KeyError: store_id 9999 not found in encoder
```

**Cause:** Test set có store/SKU không có trong training

**Solution:**
```python
# Use handle_unknown parameter
encoder = MEstimateEncoder(handle_unknown='value', handle_missing='value')
```

#### Error 4: Negative Predictions

**Cause:** Một số predictions có thể âm sau inverse transform

**Solution:**
```python
# Apply absolute value
predictions = np.abs(predictions)
```

### 10.2 Data Quality Checks

```python
def validate_input(df):
    """Comprehensive input validation"""
    errors = []
    
    # Check columns
    required_cols = ['record_ID', 'week', 'store_id', 'sku_id', 
                     'total_price', 'base_price', 'is_featured_sku', 'is_display_sku']
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # Check data types
    if df['store_id'].dtype not in ['int64', 'int32']:
        errors.append("store_id must be integer")
    
    # Check value ranges
    if (df['is_featured_sku'].notna() & ~df['is_featured_sku'].isin([0, 1])).any():
        errors.append("is_featured_sku must be 0 or 1")
    
    # Check duplicates
    if df['record_ID'].duplicated().any():
        errors.append("Duplicate record_IDs found")
    
    # Check missing
    critical_missing = df[['week', 'store_id', 'sku_id', 'base_price']].isnull().sum()
    if critical_missing.any():
        errors.append(f"Missing critical values: {critical_missing[critical_missing > 0]}")
    
    return errors

# Usage
errors = validate_input(test_df)
if errors:
    for err in errors:
        print(f"❌ {err}")
else:
    print("✓ Validation passed!")
```

### 10.3 Performance Issues

**Problem:** Inference quá chậm

**Solutions:**
1. **Use fewer models:** Dùng 3-5 models thay vì 10
2. **Batch processing:** Process nhiều samples cùng lúc
3. **Cache encoders:** Load encoders một lần, reuse nhiều lần
4. **Optimize feature engineering:** Vectorize operations

```python
# Slow
for i, row in df.iterrows():
    df.loc[i, 'diff'] = row['base_price'] - row['total_price']

# Fast (vectorized)
df['diff'] = df['base_price'] - df['total_price']
```

### 10.4 Memory Issues

**Problem:** Out of memory với large dataset

**Solutions:**
1. **Chunk processing:** Read CSV in chunks
2. **Reduce dtypes:** Use float32 instead of float64
3. **Delete unused columns:** Drop columns after use

```python
# Memory optimization
df = df.astype({
    'base_price': 'float32',
    'total_price': 'float32',
    'store_id': 'int16',
    'sku_id': 'int32'
})
```

---

## 11. Appendix

### 11.1 Glossary

| Term | Definition |
|------|------------|
| **SKU** | Stock Keeping Unit - unique identifier cho sản phẩm |
| **Store** | Cửa hàng / điểm bán |
| **Week** | Tuần bắt đầu từ ngày được chỉ định |
| **Units Sold** | Số lượng sản phẩm đã bán |
| **Base Price** | Giá gốc của sản phẩm |
| **Total Price** | Giá bán thực tế (sau discount) |
| **Featured SKU** | Sản phẩm được highlight/quảng cáo |
| **Display SKU** | Sản phẩm được display ở vị trí đặc biệt |
| **RMSLE** | Root Mean Squared Logarithmic Error - metric đánh giá |
| **DART** | Dropouts meet Multiple Additive Regression Trees - LightGBM boosting type |
| **M-Estimate** | Target encoding technique với regularization |
| **Ensemble** | Kết hợp nhiều models để improve prediction |

### 11.2 Reference Links

- **LightGBM Documentation:** https://lightgbm.readthedocs.io/
- **Category Encoders:** http://contrib.scikit-learn.org/category_encoders/
- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **RMSLE Metric:** https://www.kaggle.com/c/demand-forecasting-kernels-only/overview/evaluation

### 11.3 Model Artifacts

**Required Files:**
```
project/
├── models/
│   ├── model_fold_0.txt
│   ├── model_fold_1.txt
│   ├── ...
│   └── model_fold_9.txt
├── encoders/
│   ├── store_encoder.pkl
│   ├── sku_encoder.pkl
│   └── time_encoder.pkl
├── config/
│   └── feature_columns.json
└── inference.py
```

### 11.4 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 2024 | Initial release với 10-fold CV |
| 1.1 | - | Planned: Hyperparameter tuning |
| 1.2 | - | Planned: Feature selection |

### 11.5 Contact & Support
---

## 📝 Summary Checklist

Trước khi chạy inference, đảm bảo:

- [ ] Python 3.7+ installed
- [ ] All required libraries installed (`pip install -r requirements.txt`)
- [ ] Test data format correct (CSV với 8 columns)
- [ ] All model files available (10 fold models + 3 encoders)
- [ ] Week column theo format DD/MM/YY
- [ ] No missing values trong critical columns
- [ ] Output directory có write permission

**Expected Runtime:** 4-5 giây cho 13,860 samples

**Expected Output:** CSV file với 2 columns (record_ID, units_sold)

---

**END OF DOCUMENT**

*Last updated: November 2025*

