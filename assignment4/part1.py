import numpy as np
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
x_train_new=pca(x_train_new,5)
x_test=pca(x_test,5)
x_val=pca(x_val,5)

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column >= self.threshold] = -1
        return predictions

class Adaboost:
    def __init__(self, n_clf=300):
        self.n_clf = n_clf
        self.clfs = []
        self.val_accuracies = []

    def fit(self, X_train, y_train, X_val, y_val):
        n_samples, n_features = X_train.shape
        self.weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = X_train[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    # Predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = self.weights[y_train != predictions]
                    error = np.sum(misclassified)
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # Store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # Calculate alpha
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # Calculate predictions and update weights
            predictions = clf.predict(X_train)
            self.weights *= np.exp(-clf.alpha * y_train * predictions)
            self.weights /= np.sum(self.weights)

            val_preds = self.predict(X_val)
            val_accuracy = np.mean(val_preds == y_val)
            self.val_accuracies.append(val_accuracy)
            print(f"Iteration {len(self.clfs) + 1}: Val Accuracy = {val_accuracy:.4f}")

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

    def plot_accuracy(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.n_clf + 1), self.val_accuracies)
        plt.xlabel("Number of Estimators")
        plt.ylabel("Validation Accuracy")
        plt.title("AdaBoost Validation Accuracy")
        plt.grid()
        plt.show()

# Usage
n_clf = 300
model = Adaboost(n_clf)
model.fit(x_train_new, y_train_new, x_val, y_val)
model.plot_accuracy()
test_preds = model.predict(x_test)
test_accuracy = np.mean(test_preds == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
