import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

n = sys.argv[1]
pt = sys.argv[2]

test_data = pd.read_csv(f'../dataset/{pt}/final_test.csv')
test_data = test_data.dropna()
test_ids = test_data['ID']
X_test = test_data.drop(['isCheater', 'ID'], axis=1)
y_test = test_data['isCheater']

# Standardize the data
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Load the best model from disk
model_save_path = os.path.join(script_dir, f'{pt}/models/LogisticRegression_best_model_{n}.pkl')
model = joblib.load(model_save_path)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.5f}")

# Calculate F1 score
f1 = f1_score(y_test, predictions)
print(f"F1 score: {f1:.5f}")

# Calculate precision
precision = precision_score(y_test, predictions)
print(f"Precision: {precision:.5f}")

# Calculate recall
recall = recall_score(y_test, predictions)
print(f"Recall: {recall:.5f}")

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
fpr = fp/(fp+tn)
print(f"FPR: {fpr:.5f}")

fnr = fn/(fn+tp)
print(f"FNR: {fnr:.5f}")

# Write results to file
result_txt_path = os.path.join(script_dir, f'{pt}/test', f'LogisticRegression_{n}.txt')
create_directory_if_not_exists(os.path.dirname(result_txt_path))
with open(result_txt_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.5f}\n")
    f.write(f"F1 score: {f1:.5f}\n")
    f.write(f"Precision: {precision:.5f}\n")
    f.write(f"Recall: {recall:.5f}\n")

# Generate confusion matrix
conf_matrix_path = os.path.join(script_dir, f'{pt}/results', 'results_test', f'LR_{n}.pdf')
create_directory_if_not_exists(os.path.dirname(conf_matrix_path))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, predictions), display_labels=['Honest', 'Dishonest'])
cm_display.plot()
plt.title('LR_confusion_matrix')
plt.savefig(conf_matrix_path)
plt.show()

# Create a DataFrame with predictions, true labels, and IDs
pred_df = pd.DataFrame({'ID': test_ids, 'Predicted': predictions, 'True Label': y_test})

# Write the DataFrame to CSV
pred_csv_path = os.path.join(script_dir, f'{pt}/test', f'LogisticRegression_test_pred_{n}.csv')
create_directory_if_not_exists(os.path.dirname(pred_csv_path))
pred_df.to_csv(pred_csv_path, index=False)
