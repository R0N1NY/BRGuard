import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
# X_test = test_data.drop('ID', axis=1)
y_test = test_data['isCheater']

# Standardize the data
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Load the best model from disk
model_save_path = os.path.join(script_dir, f'{pt}/models/RandomForest_best_model_{n}.pkl')
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

# Write results to file
result_txt_path = os.path.join(script_dir, f'{pt}/test', f'RandomForest_{n}.txt')
create_directory_if_not_exists(os.path.dirname(result_txt_path))
with open(result_txt_path, 'w') as f:
    # f.write(f"Best Model Parameters: {loaded_model.get_params()}\n")
    f.write(f"Accuracy: {accuracy:.5f}\n")
    f.write(f"F1 score: {f1:.5f}\n")
    f.write(f"Precision: {precision:.5f}\n")
    f.write(f"Recall: {recall:.5f}\n")

# Generate confusion matrix
conf_matrix_path = os.path.join(script_dir, f'{pt}/results', 'results_test', f'RF_{n}.pdf')
create_directory_if_not_exists(os.path.dirname(conf_matrix_path))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, predictions), display_labels=['Honest', 'Dishonest'])
cm_display.plot()
plt.title('RF_confusion_matrix')
plt.savefig(conf_matrix_path)
plt.show()
# plt.savefig('RF.png', dpi=3600, format='png')

# Create a DataFrame with predictions, true labels, and IDs
pred_df = pd.DataFrame({'ID': test_ids, 'Predicted': predictions, 'True Label': y_test})

# Write the DataFrame to CSV
pred_csv_path = os.path.join(script_dir, f'{pt}/test', f'RandomForest_test_pred_{n}.csv')
create_directory_if_not_exists(os.path.dirname(pred_csv_path))
pred_df.to_csv(pred_csv_path, index=False)
