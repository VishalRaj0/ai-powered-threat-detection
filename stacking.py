import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ------------------ Load and Prepare Data ------------------
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "target", "extra"
]

# Load Data
train_df = pd.read_csv("nsl-kdd/KDDTrain+.txt", names=col_names).iloc[:, :-1]
test_df = pd.read_csv("nsl-kdd/KDDTest+.txt", names=col_names).iloc[:, :-1]

# Binary labels
train_df['label'] = train_df['target'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['target'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categoricals
categorical_cols = train_df.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
    test_df[col] = le.fit_transform(test_df[col])

# Split into features and target
X_train = train_df.drop(['label', 'target'], axis=1)
y_train = train_df['label']
X_test = test_df.drop(['label', 'target'], axis=1)
y_test = test_df['label']

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------ Handle Class Imbalance using SMOTE ------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# ------------------ Hyperparameter Tuning ------------------

# RandomForest Classifier
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf_clf, rf_params, cv=5, n_jobs=-1, verbose=1)
rf_grid.fit(X_train_resampled, y_train_resampled)
best_rf_clf = rf_grid.best_estimator_

# XGBoost Classifier
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_clf = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_grid = GridSearchCV(xgb_clf, xgb_params, cv=5, n_jobs=-1, verbose=1)
xgb_grid.fit(X_train_resampled, y_train_resampled)
best_xgb_clf = xgb_grid.best_estimator_

# Logistic Regression Classifier
lr_params = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

lr_clf = LogisticRegression(max_iter=1000)
lr_grid = GridSearchCV(lr_clf, lr_params, cv=5, n_jobs=-1, verbose=1)
lr_grid.fit(X_train_resampled, y_train_resampled)
best_lr_clf = lr_grid.best_estimator_

# ------------------ Stacking Classifier ------------------
# Using Logistic Regression as meta-model
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', best_rf_clf),
        ('xgb', best_xgb_clf),
        ('lr', best_lr_clf)
    ],
    final_estimator=LogisticRegression()
)

stacking_clf.fit(X_train_resampled, y_train_resampled)

# ------------------ Evaluate the Model ------------------

y_pred = stacking_clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


import joblib

# Save the trained model
joblib.dump(stacking_clf, 'stacking_model.pkl')

# # Load the trained model
# stacking_clf_loaded = joblib.load('stacking_model.pkl')
#
# # Now you can use the loaded model for predictions
# y_pred = stacking_clf_loaded.predict(X_test_scaled)
