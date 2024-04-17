import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(16)
data=np.load('mnist.npz')
lst=(data.files)
x_test=list(data[lst[0]])
x_train=list(data[lst[1]])
y_train=list(data[lst[2]])
y_test=list(data[lst[3]])
#0 to -1
for i in range(len(y_train)):
    if y_train[i]==0:
        y_train[i]=-1
for i in range(len(y_test)):
    if y_test[i]==0:
        y_test[i]=-1
#cont
for i in range(len(x_train)):
    x_train[i]=x_train[i].flatten()
x_train=np.array(x_train)
for i in range(len(x_test)):
    x_test[i]=x_test[i].flatten()
x_test=np.array(x_test)
y_test=np.array(y_test)
y_train=np.array(y_train)

#selcting classes-> 0 and 1
x_train=x_train[np.isin(y_train,[-1,1])]
y_train=y_train[np.isin(y_train,[-1,1])]
x_test=x_test[np.isin(y_test,[-1,1])]
y_test=y_test[np.isin(y_test,[-1,1])]
idx_0 = np.where(y_train == -1)[0]
idx_1 = np.where(y_train == 1)[0]
x_val = np.concatenate([x_train[idx_0[:1000]], x_train[idx_1[:1000]]], axis=0)
y_val = np.concatenate([y_train[idx_0[:1000]], y_train[idx_1[:1000]]], axis=0)
x_train_new = np.concatenate([x_train[idx_0[1000:]], x_train[idx_1[1000:]]], axis=0)
y_train_new = np.concatenate([y_train[idx_0[1000:]], y_train[idx_1[1000:]]], axis=0)
# print(x_train_new.shape)
# print(y_train_new.shape)
# print(x_val.shape)
# print(y_val.shape)
def pca(X,compnum):
    X=np.array(X)
    mu=np.mean(X,keepdims=True, axis=1)
    X_std=X-mu
    covariancematrix=np.matmul(X_std.T,X_std)
    eval,evec=np.linalg.eigh(covariancematrix)
    sorted_eig=np.argsort(-eval)
    evec=evec[:,sorted_eig]
    U=evec[:,range(compnum)]
    P=X_std.dot(U)
    return P
x_train=pca(x_train,5)
x_test=pca(x_test,5)

class DecisionStump:
    def __init__(self):
        self.split_dim = None
        self.split_val = None
        self.left = None
        self.right = None
        self.alpha = None

    def fit(self, x, y, sample_weights):
        num_features = x.shape[1]
        best_error = np.inf

        for dim in range(num_features):
            unique_values = np.unique(x[:, dim])
            unique_splits = (unique_values[:-1] + unique_values[1:]) / 2

            for split_val in unique_splits:
                predictions = np.ones_like(y)
                predictions[x[:, dim] < split_val] = -1
                error = np.sum(sample_weights[y != predictions])

                if error < best_error:
                    best_error = error
                    self.split_dim = dim
                    self.split_val = split_val
                    self.left = predictions.copy()
                    self.right = -predictions.copy()

    def predict(self, x):
        predictions = np.ones(len(x))
        predictions[x[:, self.split_dim] < self.split_val] = -1
        return predictions

class DecisionStump:
    def __init__(self):
        self.split_dim = None
        self.split_val = None
        self.left = None
        self.right = None
        self.alpha = None

    def fit(self, x, y, sample_weights):
        num_features = x.shape[1]
        best_error = np.inf

        for dim in range(num_features):
            if dim < 5:  # Only consider the first 5 dimensions
                unique_values = np.unique(x[:, dim])
                unique_splits = (unique_values[:-1] + unique_values[1:]) / 2

                for split_val in unique_splits:
                    predictions = np.ones_like(y)
                    predictions[x[:, dim] < split_val] = -1
                    error = np.sum(sample_weights[y != predictions])

                    if error < best_error:
                        best_error = error
                        self.split_dim = dim
                        self.split_val = split_val
                        self.left = predictions.copy()
                        self.right = -predictions.copy()

    def predict(self, x):
        predictions = np.ones(len(x))
        if self.split_dim is not None and self.split_val is not None:
            predictions[x[:, self.split_dim] < self.split_val] = -1
        return predictions
def boosting(x_train, y_train, x_val, y_val, num_iterations):
    num_samples = len(y_train)
    sample_weights = np.ones(num_samples) / num_samples
    classifiers = []
    val_accuracies = []

    for i in range(num_iterations):
        stump = DecisionStump()
        stump.fit(x_train, y_train, sample_weights)
        predictions = stump.predict(x_train)

        weighted_error = np.sum(sample_weights[y_train != predictions])
        alpha_i = 0.5 * np.log((1 - weighted_error) / weighted_error)

        sample_weights *= np.exp(-y_train * predictions * alpha_i)
        sample_weights /= np.sum(sample_weights)

        classifiers.append((stump, alpha_i))

        # Evaluate on validation set
        val_predictions = np.sum([alpha * clf.predict(x_val) for clf, alpha in classifiers], axis=0)
        val_accuracy = np.mean(np.sign(val_predictions) == y_val)
        val_accuracies.append(val_accuracy)

    return classifiers, val_accuracies
# Running boosting algorithm
num_iterations = 300
classifiers, val_accuracies = boosting(x_train_new, y_train_new, x_val, y_val, num_iterations)

# Plot validation accuracies vs number of trees
plt.plot(range(1, num_iterations + 1), val_accuracies)
plt.xlabel('Number of Trees')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Number of Trees')
plt.show()

# Select the best performing classifier
best_idx = np.argmax(val_accuracies)
best_classifier, _ = classifiers[best_idx]

# Evaluate the selected classifier on the test set
test_predictions = np.zeros(len(x_test))
for clf, alpha in classifiers[:best_idx + 1]:
    test_predictions += alpha * clf.predict(x_test)
test_accuracy = np.mean(np.sign(test_predictions) == y_test)
print(val_accuracies)
print("Test Accuracy:", test_accuracy)
