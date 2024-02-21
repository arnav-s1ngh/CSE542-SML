import numpy as np
import matplotlib.pyplot as plt
#preprocessing

data = np.load('mnist.npz')
lst =(data.files)
x_test=list(data[lst[0]])
x_train=list(data[lst[1]])
y_train=list(data[lst[2]])
y_test=list(data[lst[3]])
# print(type(x_test))
# print(type(lst))
# for i in lst:
#     print(i)
#     print(data[i])

#vectorisation
for i in range(len(x_train)):
    x_train[i] = x_train[i].flatten()
x_train=np.array(x_train)


for i in range(len(x_test)):
    x_test[i] = x_test[i].flatten()
x_test=np.array(x_test)
X=[]
X_test=[]
Y_train=[]
Y_test=[]
for i in range(10):
    class_lst=[dat for dat in range(len(y_train)) if y_train[dat]==i]
    for j in range(100):
        X.append(x_train[j])
        Y_train.append(i)
for i in range(1):
    class_lst=[dat for dat in range(len(y_test)) if y_test[dat]==i]
    for j in range(100):
        X_test.append(x_test[j])
        Y_test.append(i)
X_test=np.array(X_test)
X_test=X_test.T
X_test_centered=X_test-np.mean(X_test,keepdims=True,axis=1)
X=np.array(X)
X=X.T
X_centered = X - np.mean(X,keepdims=True, axis=1)
later=np.mean(X,keepdims=True, axis=1)
print(X_centered.shape)
S = np.dot(X_centered,X_centered.T)/999
print(S.shape)
eigenvalues, eigenvectors = np.linalg.eigh(S)
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[i,:]) for i in range(len(eigenvalues))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
U = np.array([x[1] for x in eig_pairs])

Y = np.dot(U.T, X_centered)

X_reconstructed = np.dot(U, Y)

MSE = np.mean((X - X_reconstructed) ** 2,axis=1)
print(MSE)

for p in [500,600,700,800,900,1000]:
    Up = U[:, :p]
    Yp = np.dot(Up.T, X_centered)  # Reduced feature matrix
    X_reconstructed=np.dot(Up, Yp)
    image = np.reshape((X_reconstructed+later)[:, 1], (28, 28))
    plt.imshow(image, cmap='gray')
    plt.title(f'{p})')
    plt.show()



for p in [10000000]:
    Up = U[:, :p]
    Yp_train=np.dot(Up.T, X_centered)
    Yp_test = np.dot(Up.T, X_test_centered)  
    print("yp",Yp_test.shape)
    arr_mean = []
    arr_cov = []
    for i in range(10):
        class_lst = [w for w in range(len(Y_train)) if Y_train[w] == i]
        local_x = Yp_train.T[class_lst]
        mat = np.cov(local_x, rowvar=False)
        arr_cov.append(mat)
        arr_mean.append(np.mean(local_x, axis=0))
    arr_mean = np.array(arr_mean)
    arr_cov = np.array(arr_cov)
    inv_covariances = np.linalg.pinv(arr_cov)
    log_determinants = np.log(np.linalg.det(arr_cov) + 0.000001)
    prior = []
    for i in range(10):
        su = 0
        for j in Y_train:
            if j == i:
                su += 1
        prior.append(su / len(Y_train))
    predicted_classes = []
    for sample in Yp_test.T:
        probabilities = np.zeros(10)  
        for i in range(10):
            diff = sample - arr_mean[i]
            quadratic_term = -0.5 * np.dot(diff.T, np.dot(inv_covariances[i], diff))
            probabilities[i] = quadratic_term - 0.5 * log_determinants[i] + np.log(prior[i])
        predicted_class = np.argmax(probabilities)
        predicted_classes.append(predicted_class)
    class_accuracy = []
    for i in range(10):
        class_lst = np.array([w for w in range(len(predicted_classes)) if predicted_classes[w] == i])
        su = 0
        for j in range(len(class_lst)):
            if predicted_classes[j] == y_test[j]:
                su += 1
        class_accuracy.append(su / ((len(class_lst))+1))
    print("Class-wise Accuracy:", class_accuracy)
    print("Total Accuracy:",sum(class_accuracy)/len(class_accuracy))
