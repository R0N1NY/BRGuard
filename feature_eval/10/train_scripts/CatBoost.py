import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import sys
import os

default_num_threads = 2

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

n = sys.argv[1]
pt = sys.argv[2]
num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else default_num_threads

data = pd.read_csv(f'../dataset/{pt}/final_train_{n}.csv')
data = data.dropna()
train_ids = data['ID']
X_train = data.drop(['isCheater', 'ID'], axis=1)
y_train = data['isCheater']

test_data = pd.read_csv(f'../dataset/{pt}/final_val.csv')
test_data = test_data.dropna()
test_ids = test_data['ID']
X_test = test_data.drop(['isCheater', 'ID'], axis=1)
y_test = test_data['isCheater']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

test_fold = [-1] * len(X_train) + [0] * len(X_test)
ps = PredefinedSplit(test_fold)

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'iterations': [100, 500, 1000],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64, 128],
    'loss_function': ['Logloss', 'CrossEntropy']
}

model = CatBoostClassifier(verbose=False, thread_count=num_threads, task_type='CPU')

grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=ps, verbose=2)
grid_search.fit(X_combined, y_combined)

best_model = grid_search.best_estimator_

model_save_path = os.path.join(script_dir, f'{pt}/models/CatBoost_best_model_{n}.pkl')
create_directory_if_not_exists(os.path.dirname(model_save_path))
joblib.dump(best_model, model_save_path)

loaded_model = joblib.load(model_save_path)

predictions = loaded_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.5f}")

f1 = f1_score(y_test, predictions)
print(f"F1 score: {f1:.5f}")

precision = precision_score(y_test, predictions)
print(f"Precision: {precision:.5f}")

recall = recall_score(y_test, predictions)
print(f"Recall: {recall:.5f}")

train_result_path = os.path.join(script_dir, f'{pt}/train/CatBoost_{n}.txt')
create_directory_if_not_exists(os.path.dirname(train_result_path))
with open(train_result_path, 'w') as f:
    f.write(f"Best Model Parameters: {loaded_model.get_params()}\n")
    f.write(f"Accuracy: {accuracy:.5f}\n")
    f.write(f"F1 score: {f1:.5f}\n")
    f.write(f"Precision: {precision:.5f}\n")
    f.write(f"Recall: {recall:.5f}\n")

result_save_path = os.path.join(script_dir, f'{pt}/results/CatBoost_confusion_matrix_{n}.pdf')
create_directory_if_not_exists(os.path.dirname(result_save_path))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, predictions), display_labels=['Honest', 'Dishonest'])
cm_display.plot()
plt.title('CatBoost_confusion_matrix')
plt.savefig(result_save_path)
plt.show()

pred_df = pd.DataFrame({'ID': test_ids, 'Predicted': predictions, 'True Label': y_test})

pred_save_path = os.path.join(script_dir, f'{pt}/val/CatBoost_pred_{n}.csv')
create_directory_if_not_exists(os.path.dirname(pred_save_path))
pred_df.to_csv(pred_save_path, index=False)

train_predictions = loaded_model.predict(X_train)

train_df = pd.DataFrame({'ID': train_ids, 'Predicted': train_predictions, 'True Label': y_train})

train_save_path = os.path.join(script_dir, f'{pt}/train/CatBoost_train_{n}.csv')
create_directory_if_not_exists(os.path.dirname(train_save_path))
train_df.to_csv(train_save_path, index=False)
