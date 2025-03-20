
# import packages for svm
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('data.csv', header=None)
labels = pd.read_csv('labels.csv', header=None)

X = data.values
Y = labels.values

print("Data shape:", X.shape)
print("Labels shape:", Y.shape)

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# A soft (C=0:01) linear SVM trained
model = SVC(kernel='sigmoid', gamma=8e-3, C=1.0)  
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed Report
print(classification_report(y_test, y_pred))

# Predict decision scores
decision_values = model.decision_function(X_test)

# Convert to confidence values
confidence = 1.0 / (1.0 + np.exp(-decision_values))

print("Decision Scores:", decision_values)
print("Y test:", y_test)
print("Confidence Values:", confidence)

# implement precision recall curve
precision = []
recall = []
thresholds = np.unique(confidence)
for threshold in thresholds:
    y_pred = confidence > threshold
    tp = np.sum(y_test[y_pred])
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_test) - tp
    if tp + fp == 0:
        precision.append(1)
    else:
        precision.append(tp / (tp + fp))
    recall.append(tp / (tp + fn))

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve for SVM')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()

