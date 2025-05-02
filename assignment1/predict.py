import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
data = data.drop(columns=['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'])

nominal_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
data = pd.get_dummies(data, columns=nominal_columns, drop_first=False)

data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

X = data.drop('Attrition', axis=1)
y = data['Attrition']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

precision_list = []
recall_list = []
f1_list = []
auc_list = []

for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    xgb_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.8).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f'\n== Fold {fold_index} == ')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'AUC: {auc:.4f}')

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    auc_list.append(auc)

print('\n=== Overall Performance ===')
print(f'Average Precision: {np.mean(precision_list):.4f}')
print(f'Average Recall: {np.mean(recall_list):.4f}')
print(f'Average F1-Score: {np.mean(f1_list):.4f}')
print(f'Average AUC: {np.mean(auc_list):.4f}')

