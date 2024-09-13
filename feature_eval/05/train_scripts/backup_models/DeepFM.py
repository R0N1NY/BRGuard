import pandas as pd
import numpy as np
import tensorflow as tf
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import sys

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

n = sys.argv[1]
pt = sys.argv[2]

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

dense_feats = [col for col in data.columns if col not in ['ID', 'isCheater'] and data[col].dtype != 'object']
sparse_feats = [col for col in data.columns if col in ['ID', 'isCheater'] or data[col].dtype == 'object']

# Preprocessing
for feat in dense_feats:
    scaler = MinMaxScaler()
    data[feat] = scaler.fit_transform(data[[feat]])
    test_data[feat] = scaler.transform(test_data[[feat]])

for feat in sparse_feats:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
    test_data[feat] = lbe.transform(test_data[feat])

# Define feature columns for DeepFM
feature_columns = [DenseFeat(feat, 1,) for feat in dense_feats] + [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for feat in sparse_feats]

# Generate feature names
feature_names = get_feature_names(feature_columns)

# Convert pandas dataframe to model input data
train_model_input = {name: data[name] for name in feature_names}
test_model_input = {name: test_data[name] for name in feature_names}

# Define model
model = DeepFM(feature_columns, feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

# Fit model
history = model.fit(train_model_input, y_train.values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)

# Prediction
pred_ans = model.predict(test_model_input, batch_size=256)

# Evaluation
print("Test ROC AUC:", roc_auc_score(y_test, pred_ans))

# Save model - TensorFlow Keras model saving
model_save_path = os.path.join(script_dir, f'models/DeepFM_best_model_{n}.h5')
create_directory_if_not_exists(os.path.dirname(model_save_path))
model.save(model_save_path)

binary_predictions = [1 if x > 0.5 else 0 for x in pred_ans.flatten()]

# Calculate metrics
accuracy = accuracy_score(y_test, binary_predictions)
f1 = f1_score(y_test, binary_predictions)
precision = precision_score(y_test, binary_predictions)
recall = recall_score(y_test, binary_predictions)

# Write results to file
train_result_path = os.path.join(script_dir, f'train/DeepFM_{n}.txt')
create_directory_if_not_exists(os.path.dirname(train_result_path))
with open(train_result_path, 'w') as f:
    f.write("Evaluation Metrics:\n")
    f.write(f"Accuracy: {accuracy:.5f}\n")
    f.write(f"F1 score: {f1:.5f}\n")
    f.write(f"Precision: {precision:.5f}\n")
    f.write(f"Recall: {recall:.5f}\n")

# Generate and save confusion matrix
result_save_path = os.path.join(script_dir, f'results/DeepFM_confusion_matrix_{n}.pdf')
create_directory_if_not_exists(os.path.dirname(result_save_path))
cm = confusion_matrix(y_test, binary_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Honest', 'Dishonest'])
fig, ax = plt.subplots()
disp.plot(ax=ax)
plt.title('DeepFM Confusion Matrix')
plt.savefig(result_save_path)
plt.close()  # Close the plot to prevent it from displaying in non-interactive environments

# Create a DataFrame with predictions, true labels, and IDs
pred_df = pd.DataFrame({'ID': test_ids, 'Predicted': binary_predictions, 'True Label': y_test.values})

# Write the DataFrame to CSV
pred_save_path = os.path.join(script_dir, f'val/DeepFM_pred_{n}.csv')
create_directory_if_not_exists(os.path.dirname(pred_save_path))
pred_df.to_csv(pred_save_path, index=False)