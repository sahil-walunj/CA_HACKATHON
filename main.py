import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Step 1: Data Loading and Preprocessing
data = pd.read_csv('data.csv')
test_data = pd.read_csv('test.csv')

# Drop respondent_id from features but keep it in the test set for submission
X = data.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y_xyz = data['xyz_vaccine']
y_seasonal = data['seasonal_vaccine']
test_ids = test_data['respondent_id']
X_test = test_data.drop(columns=['respondent_id'])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 2: Model Building and Training
# Split the data
X_train, X_val, y_train_xyz, y_val_xyz, y_train_seasonal, y_val_seasonal = train_test_split(
    X, y_xyz, y_seasonal, test_size=0.2, random_state=42)

# Define the models
model_xyz = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

model_seasonal = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Hyperparameter tuning (example for RandomForest, adjust as needed)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

search_xyz = GridSearchCV(model_xyz, param_grid, cv=5, scoring='roc_auc')
search_seasonal = GridSearchCV(model_seasonal, param_grid, cv=5, scoring='roc_auc')

search_xyz.fit(X_train, y_train_xyz)
search_seasonal.fit(X_train, y_train_seasonal)

# Step 3: Model Evaluation
best_model_xyz = search_xyz.best_estimator_
best_model_seasonal = search_seasonal.best_estimator_

# Predictions on validation set
y_val_pred_xyz = best_model_xyz.predict_proba(X_val)[:, 1]
y_val_pred_seasonal = best_model_seasonal.predict_proba(X_val)[:, 1]

# Calculate ROC AUC
roc_auc_xyz = roc_auc_score(y_val_xyz, y_val_pred_xyz)
roc_auc_seasonal = roc_auc_score(y_val_seasonal, y_val_pred_seasonal)
mean_roc_auc = np.mean([roc_auc_xyz, roc_auc_seasonal])

print(f"Mean ROC AUC: {mean_roc_auc}")

# Step 4: Prediction and Submission
# Predictions on test set
test_pred_xyz = best_model_xyz.predict_proba(X_test)[:, 1]
test_pred_seasonal = best_model_seasonal.predict_proba(X_test)[:, 1]

# Prepare submission file
submission = pd.DataFrame({
    'respondent_id': test_ids,
    'xyz_vaccine': test_pred_xyz,
    'seasonal_vaccine': test_pred_seasonal
})

submission.to_csv('submission.csv', index=False)
