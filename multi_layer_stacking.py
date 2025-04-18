import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load NSL-KDD Dataset
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

train_df = pd.read_csv("nsl-kdd/KDDTrain+.txt", names=col_names).iloc[:, :-1]
test_df = pd.read_csv("nsl-kdd/KDDTest+.txt", names=col_names).iloc[:, :-1]

train_df['label'] = train_df['target'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['target'].apply(lambda x: 0 if x == 'normal' else 1)

full_df = pd.concat([train_df, test_df])
full_df.drop('target', axis=1, inplace=True)

for col in full_df.select_dtypes(include='object').columns:
    full_df[col] = LabelEncoder().fit_transform(full_df[col])

train_df = full_df[:len(train_df)]
test_df = full_df[len(train_df):]

X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'poly'], 'gamma': ['scale', 'auto']}
param_grid_tree = {'max_depth': [3, 5, 10]}
param_grid_mlp = {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}

svm_search = GridSearchCV(SVC(probability=True), param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1)
svm_search.fit(X_train_res, y_train_res)
best_svm = svm_search.best_estimator_

# Use default for now
sgd = SGDClassifier(random_state=42)
gnb = GaussianNB()

tree_search = GridSearchCV(DecisionTreeClassifier(), param_grid_tree, cv=3, scoring='accuracy', n_jobs=-1)
tree_search.fit(X_train_res, y_train_res)
best_tree = tree_search.best_estimator_

mlp_search = GridSearchCV(MLPClassifier(max_iter=1000), param_grid_mlp, cv=3, scoring='accuracy', n_jobs=-1)
mlp_search.fit(X_train_res, y_train_res)
best_mlp = mlp_search.best_estimator_

# Stack 1: GNB + SGD -> SVM
stack_svm = StackingClassifier(
    estimators=[("gnb", gnb), ("sgd", sgd)],
    final_estimator=best_svm,
    cv=5
)

# Stack 2: GNB + SGD -> DecisionTree
stack_tree = StackingClassifier(
    estimators=[("gnb", gnb), ("sgd", sgd)],
    final_estimator=best_tree,
    cv=5
)

class StackerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stack_model):
        self.stack_model = stack_model
    def fit(self, X, y):
        self.stack_model.fit(X, y)
        return self
    def transform(self, X):
        return self.stack_model.predict_proba(X)

stacker1 = StackerTransformer(stack_svm)
stacker2 = StackerTransformer(stack_tree)

meta_feature_union = FeatureUnion([
    ("stack_svm", stacker1),
    ("stack_tree", stacker2)
])

final_pipeline = Pipeline([
    ("meta_features", meta_feature_union),
    ("mlp_meta", best_mlp)
])

final_pipeline.fit(X_train_res, y_train_res)
y_pred = final_pipeline.predict(X_test_scaled)
y_proba = final_pipeline.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacking (SVM + Tree) → MLP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
