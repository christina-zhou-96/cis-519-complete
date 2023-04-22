import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

# Load Data
filename = 'data/SPECTF.dat'
data = np.loadtxt(filename, delimiter=',')
X = data[:, 1:]
y = np.array([data[:, 0]]).T
n, d = X.shape

# store accuracies later
decision_tree_accuracies = []
decision_stump_accuracies = []
dt3_accuracies = []

classifiers = {
    'Decision Tree': tree.DecisionTreeClassifier(),
    'Decision Stump': tree.DecisionTreeClassifier(max_depth=1),
    '3-level Decision Tree': tree.DecisionTreeClassifier(max_depth=3)
}

# Initialize arrays to store accuracy and learning curve data
accuracies = {name: [] for name in classifiers.keys()}
learning_curve_data = {name: [] for name in classifiers.keys()}

# Perform 100 trials
for trial in range(100):
    # shuffle the data
    idx = np.arange(n)
    np.random.seed(13 + trial)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # variables to store accuracies for current trial
    current_decision_tree_accuracies = []
    current_decision_stump_accuracies = []
    current_dt3_accuracies = []

    # 10-fold cross-validation
    fold_size = n // 10
    for i in range(10):
        # Test
        X_test = X[i * fold_size:(i + 1) * fold_size]
        y_test = y[i * fold_size:(i + 1) * fold_size]

        # Train
        X_train = np.vstack((X[:i * fold_size], X[(i + 1) * fold_size:]))
        y_train = np.vstack((y[:i * fold_size], y[(i + 1) * fold_size:]))

        # Train decision tree
        clf_decision_tree = tree.DecisionTreeClassifier()
        clf_decision_tree.fit(X_train, y_train)

        # Train decision stump
        clf_decision_stump = tree.DecisionTreeClassifier(max_depth=1)
        clf_decision_stump.fit(X_train, y_train)

        # Train 3-level decision tree
        clf_dt3 = tree.DecisionTreeClassifier(max_depth=3)
        clf_dt3.fit(X_train, y_train)

        # Predict current fold
        y_pred_decision_tree = clf_decision_tree.predict(X_test)
        y_pred_decision_stump = clf_decision_stump.predict(X_test)
        y_pred_dt3 = clf_dt3.predict(X_test)

        # Calculate accuracies for the current fold
        current_decision_tree_accuracies.append(accuracy_score(y_test, y_pred_decision_tree))
        current_decision_stump_accuracies.append(accuracy_score(y_test, y_pred_decision_stump))
        current_dt3_accuracies.append(accuracy_score(y_test, y_pred_dt3))

        # Calculate learning curve data for each classifier and different training set sizes
        for train_size_percentage in range(10, 101, 10):
            train_size = int(train_size_percentage * fold_size / 100)
            X_train_sub = X_train[:train_size, :]
            y_train_sub = y_train[:train_size, :]

            for classifier_name, classifier in classifiers.items():
                clf = classifier.fit(X_train_sub, y_train_sub)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                if trial == 0 and i == 0:
                    learning_curve_data[classifier_name].append([accuracy])
                else:
                    learning_curve_data[classifier_name][train_size_percentage // 10 - 1].append(accuracy)

    # Store the mean accuracy for each classifier in the current trial
    decision_tree_accuracies.append(np.mean(current_decision_tree_accuracies))
    decision_stump_accuracies.append(np.mean(current_decision_stump_accuracies))
    dt3_accuracies.append(np.mean(current_dt3_accuracies))

accuracies = {'Decision Tree':decision_tree_accuracies,
              'Decision Stump':decision_stump_accuracies,
              '3-Level Decision Tree':dt3_accuracies}

# Calculate mean and standard deviation for accuracy and learning curve data
stats = {name: (np.mean(acc), np.std(acc)) for name, acc in accuracies.items()}
for name in learning_curve_data.keys():
    learning_curve_data[name] = np.array([np.mean(accuracies), np.std(accuracies)] for accuracies in learning_curve_data[name])

plt.figure()
for name, data in learning_curve_data.items():
    plt.errorbar(range(10, 101, 10), data[:, 0], yerr=data[:, 1], label=name)

plt.show()

# Calculate mean and standard deviation for each classifier
mean_decision_tree_accuracy = np.mean(decision_tree_accuracies)
stddev_decision_tree_accuracy = np.std(decision_tree_accuracies)
mean_decision_stump_accuracy = np.mean(decision_stump_accuracies)
stddev_decision_stump_accuracy = np.std(decision_stump_accuracies)
mean_dt3_accuracy = np.mean(dt3_accuracies)
stddev_dt3_accuracy = np.std(dt3_accuracies)

# make certain that the return value matches the API specification
stats = np.zeros((3, 2))
stats[0, 0] = mean_decision_tree_accuracy
stats[0, 1] = stddev_decision_tree_accuracy
stats[1, 0] = mean_decision_stump_accuracy
stats[1, 1] = stddev_decision_stump_accuracy
stats[2, 0] = mean_dt3_accuracy
stats[2, 1] = stddev_dt3_accuracy
