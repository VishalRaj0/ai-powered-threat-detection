import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # <-- Added for SMOTE

# ---------------------------------------------
# Step 1: Load the data
# ---------------------------------------------
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
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "target", "extra-label"
]

attack_map = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R'
}

train_df = pd.read_csv("nsl-kdd/KDDTrain+.txt", names=col_names, index_col=False).drop('extra-label', axis=1)
test_df = pd.read_csv("nsl-kdd/KDDTest+.txt", names=col_names, index_col=False).drop('extra-label', axis=1)

# ---------------------------------------------
# Step 2: Label binary classes
# ---------------------------------------------
def label_binary(x):
    return 0 if x == 'normal' else 1

train_df['label'] = train_df['target'].apply(label_binary)
test_df['label'] = test_df['target'].apply(label_binary)

train_df['attack_category'] = train_df['target'].map(attack_map)
test_df['attack_category'] = test_df['target'].map(attack_map)

train_df = train_df.drop(['target', 'attack_category'], axis=1)
test_df = test_df.drop(['target', 'attack_category'], axis=1)

# ---------------------------------------------
# Step 3: Combine for consistent encoding
# ---------------------------------------------
combined = pd.concat([train_df, test_df])
categorical_cols = combined.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

train_df = combined[:len(train_df)]
test_df = combined[len(train_df):]

# ---------------------------------------------
# Step 4: Feature scaling
# ---------------------------------------------
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- ✅ SMOTE Resampling ----------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# ---------- Step 5: Model training ----------
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# ---------- Step 6: Evaluate ----------
y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

import joblib

# Save the trained model
joblib.dump(model, 'XGBclassifier_model.pkl')