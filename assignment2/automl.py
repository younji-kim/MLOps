import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator, H2OXGBoostEstimator, H2ODeepLearningEstimator

data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
data = data.drop(columns=['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'])

nominal_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
data = pd.get_dummies(data, columns=nominal_columns, drop_first=False)
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

h2o.init(max_mem_size="2G", nthreads=-1)
hf = h2o.H2OFrame(data)
hf['Attrition'] = hf['Attrition'].asfactor()

x = hf.columns
x.remove('Attrition')
y = 'Attrition'

train, test = hf.split_frame(ratios=[0.8], seed=1)

def evaluate_h2o_model(model, train, test, x, y):
    model.train(x=x, y=y, training_frame=train)
    perf = model.model_performance(test_data=test)
    y_pred = model.predict(test).as_data_frame()['predict'].astype(int)
    y_true = test[y].as_data_frame()[y].astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = perf.auc()
    return precision, recall, f1, auc

results = []

for max_depth in [5, 10, 15]:
    model = H2OGradientBoostingEstimator(ntrees=100, max_depth=max_depth, seed=1)
    precision, recall, f1, auc = evaluate_h2o_model(model, train, test, x, y)
    results.append({
        'Model': 'GBM',
        'Param': f'max_depth={max_depth}',
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc
    })

for sample_rate in [0.6, 0.8, 1.0]:
    model = H2OXGBoostEstimator(ntrees=100, sample_rate=sample_rate, seed=1)
    precision, recall, f1, auc = evaluate_h2o_model(model, train, test, x, y)
    results.append({
        'Model': 'XGBoost',
        'Param': f'sample_rate={sample_rate}',
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc
    })

for dropout in [0.0, 0.1, 0.2]:
    model = H2ODeepLearningEstimator(epochs=100, input_dropout_ratio=dropout, seed=1)
    precision, recall, f1, auc = evaluate_h2o_model(model, train, test, x, y)
    results.append({
        'Model': 'DeepLearning',
        'Param': f'input_dropout_ratio={dropout}',
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc
    })

df_results = pd.DataFrame(results)
print("\n=== Comparison of H2O Models with Varying Hyperparameters ===")
print(df_results[['Model', 'Param', 'Precision', 'Recall', 'F1', 'AUC']].to_string(index=False))

h2o.shutdown(prompt=False)

