from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data.csv', header=None)
labels = pd.read_csv('labels.csv', header=None).values.ravel()
print("Shape of data:", data.shape)
print("Shape of labels:", labels.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the model
clf = MLPClassifier(
    hidden_layer_sizes=(100, 100), 
    activation='tanh',
    max_iter=1000, 
    learning_rate='adaptive', 
    learning_rate_init=0.0001, 
    alpha=0.001, 
    solver='adam', 
    random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)
confidence = y_pred_prob.max(axis=1)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confidence:', confidence)

# precision recall curve
precision = []
recall = []
thresholds = np.unique(confidence)
thresholds = np.sort(thresholds)[::-1]
for threshold in thresholds:
    y_pred = confidence >= threshold
    tp = np.sum(y_test[y_pred])
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_test) - tp
    # print(threshold, y_pred, type(tp), type(fp), type(fn))
    precision.append(tp / (tp + fp))
    recall.append(tp / (tp + fn))

# precision recall curve plot
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
ax.plot(recall, precision, 'o', color='black')

# zip joins x and y coordinates in pairs
for x,y,c in zip(recall, precision, thresholds):

    label = "c = " + "{:.2f}".format(c)
    
    if y == 1:
        ax.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 fontsize=7,
                 xytext=(0,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    else:
        ax.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 fontsize=7,
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='left')

# axis labels to plot
ax.set_title('Precision-Recall Curve for MLP')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

plt.show()

