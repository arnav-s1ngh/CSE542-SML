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
figure, ax = plt.subplots(10, 5, figsize=(10, 5))
# print(y_train)
# print([w for w in range(len(y_train)) if y_train[w]==7])
# pos=[w for w in range(len(y_train)) if y_train[w]==7]
# print([y_train[x] for x in pos])

#image-ify

for i in range(10):
    class_lst =[w for w in range(len(y_train)) if y_train[w]==i][0:5]
    #first 5 images in ascending order of indices for each class
    # print(class_indices)
    for j in range(5):
        mnist = x_train[class_lst[j]]
        #gist_earth>>>>>>>grey
        ax[i, j].imshow(mnist, cmap='gist_earth')
        ax[i, j].axis('off')
plt.tight_layout()
plt.show()

#vectorisation and qda
for i in range(len(x_train)):
    x_train[i] = x_train[i].flatten()
x_train=np.array(x_train)
print(x_train)


for i in range(len(x_test)):
    x_test[i] = x_test[i].flatten()
x_test=np.array(x_test)


def div_vec(vector,num):
    return [i/num for i in vector]


arr_mean=[]
arr_cov=[]

for i in range(10):
    class_lst = [w for w in range(len(y_train)) if y_train[w] == i]
    local_x=x_train[class_lst]
    mat=np.cov(local_x, rowvar=False)
    arr_cov.append(mat)
    arr_mean.append(np.mean(local_x,axis=0))

arr_mean=np.array(arr_mean)
arr_cov=np.array(arr_cov)

print(arr_mean)
print("done")
# sir's technique doesn't work, better to add a small const to det, and take pinv
icovariances = np.linalg.pinv(arr_cov)
ldeterminants = np.log(np.linalg.det(arr_cov) + 0.000001)
prior=[]
for i in range(10):
    su=0
    for j in y_train:
        if j==i:
            su+=1
    prior.append(su/len(y_train))
print(prior)
print(sum(prior))
predicted_classes = []
for sample in x_test:
    probabilities = np.zeros(10)
    for i in range(10):
        diff = sample - arr_mean[i]
        quadterm = -0.5 * np.dot(diff.T, np.dot(icovariances[i], diff))
        probabilities[i] = quadterm - 0.5 * ldeterminants[i]+np.log(prior[i])
    predicted_class = np.argmax(probabilities)
    predicted_classes.append(predicted_class)

#finding class accuracies and their weighted avg for total
class_accuracy = []
for i in range(10):
    class_lst =np.array([w for w in range(len(predicted_classes)) if predicted_classes[w] == i])
    su=0
    for j in range(len(class_lst)):
        if predicted_classes[j]==y_test[j]:
            su+=1
    class_accuracy.append(su / len(class_lst))

print("Class-wise Accuracy:", class_accuracy)
print("Overall Accuracy:", sum(class_accuracy[i]*prior[i] for i in range(10)))
