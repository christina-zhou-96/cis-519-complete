import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score

def evaluatePerformance():
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

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
        idx = np.arange(n)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # 10-fold cross-validation
        fold_size = n // 10
        for i in range(10):
            X_test = X[i * fold_size:(i + 1) * fold_size, :]
            y_test = y[i * fold_size:(i + 1) * fold_size, :]
            X_train = np.concatenate((X[:i * fold_size, :], X[(i + 1) * fold_size:, :]), axis=0)
            y_train = np.concatenate((y[:i * fold_size, :], y[(i + 1) * fold_size:, :]), axis=0)

            # Calculate accuracy and learning curve data for each classifier
            for name, clf in classifiers.items():
                clf = clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                accuracies[name].append(accuracy)

                # Calculate learning curve data for different training set sizes
                for train_size_percentage in range(10, 101, 10):
                    train_size = int(train_size_percentage * fold_size / 100)
                    X_train_sub = X_train[:train_size, :]
                    y_train_sub = y_train[:train_size, :]

                    clf = clf.fit(X_train_sub, y_train_sub)
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    if trial == 0 and i == 0:
                        learning_curve_data[name].append([accuracy])
                    else:
                        learning_curve_data[name][train_size_percentage // 10 - 1].append(accuracy)

    # Calculate mean and standard deviation for accuracy and learning curve data
    stats = {name: (np.mean(acc), np.std(acc)) for name, acc in accuracies.items()}
    for name in learning_curve_data.keys():
        learning_curve_data[name] = np.array([np.mean(accuracies), np.std(accuracies)] for accuracies in learning_curve_data[name])

    return stats, learning_curve_data

def plot_learning_curve(learning_curve_data):
    plt.figure()
    for name, data in learning_curve_data.items():
        plt.errorbar(range(10, 101, 10), data[:, 0], yerr=data[:, 1], label=name)

    plt.xlabel('Training Data Percentage')
    plt.ylabel('Test Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    stats, learning_curve_data = evaluatePerformance()
    plot_learning_curve(learning_curve_data)