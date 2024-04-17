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
# for i in range(len(y_train)):
#     if y_train[i]==0:
#         y_train[i]=-1
# for i in range(len(y_test)):
#     if y_test[i]==0:
#         y_test[i]=-1
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
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def split(self, X, y):
        n_samples, n_features = X.shape
        min_ssr = float('inf')
        best_feature = None
        best_threshold = None
        for feature_i in range(n_features):
            X_column = X[:, feature_i]
            thresholds = np.unique(X_column)
            thresholds.sort()
            split_points = [(thresholds[i] + thresholds[i-1]) / 2 for i in range(1, len(thresholds))]
            for threshold in split_points:
                left = y[X_column < threshold]
                right = y[X_column >= threshold]
                ssr = np.sum((left - np.mean(left))**2) + np.sum((right - np.mean(right))**2)
                if ssr < min_ssr:
                    min_ssr = ssr
                    best_feature = feature_i
                    best_threshold = threshold
        self.feature_idx = best_feature
        self.threshold = best_threshold
        return min_ssr

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.zeros(n_samples, dtype=float)
        predictions[X_column < self.threshold] = -0.01
        predictions[X_column >= self.threshold] = 0.01
        return predictions

num_trees = 300
trees = []
residuals = y_train_new.copy().astype(float)

for t in range(num_trees):
    stump = DecisionStump()
    ssr = stump.split(x_train_new, residuals)
    stump.alpha = 1
    trees.append(stump)
    residuals -= stump.alpha * stump.predict(x_train_new)

    # Evaluate on the validation set
    val_pred = np.sum([tree.alpha * tree.predict(x_val) for tree in trees], axis=0)
    val_mse = np.mean((y_val - val_pred) ** 2)
    print(f"Iteration {t}, Validation MSE: {val_mse:.4f}")

# Evaluate on the test set
test_pred = np.sum([tree.alpha * tree.predict(x_test) for tree in trees], axis=0)
test_mse = np.mean((y_test - test_pred) ** 2)
print(f"Test MSE: {test_mse:.4f}")

# Plot the validation MSE
plt.figure(figsize=(10, 6))
val_preds = [np.sum([tree.alpha * tree.predict(x_val) for tree in trees[:i+1]], axis=0) for i in range(num_trees)]
plt.plot(range(1, num_trees + 1), [np.mean((y_val - pred) ** 2) for pred in val_preds])
plt.xlabel("Number of Trees")
plt.ylabel("Validation MSE")
plt.title("Validation MSE vs Number of Trees")
plt.show()
