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
        sample_image = x_train[class_lst[j]]
        #gist_earth>>>>>>>grey
        ax[i, j].imshow(sample_image, cmap='gist_earth')
        ax[i, j].axis('off')
plt.tight_layout()
plt.show()

#vectorisation and qda
for i in range(len(x_train)):
    x_train[i] = x_train[i].flatten()
x_train=np.array(x_train)
print(x_train)


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

# discriminants = {}
# for c in range(10):
#     cov_inv = np.linalg.inv(arr_cov[c])
#     mean_diff = x_test - arr_mean[c]
#     discriminants[c] = -0.5 * np.sum(mean_diff.dot(cov_inv) * mean_diff, axis=1) - 0.5 * np.log(np.linalg.det(arr_cov[c])) + np.log(priors[c])


def qda(x, mean, cov):
    cov += np.eye(cov.shape[0]) * 0.00000000000001
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    x_minus_mean = x - mean
    x_minus_mean = x_minus_mean[np.newaxis, :]
    exponent_term = np.sum(np.dot(x_minus_mean, inv_cov) * x_minus_mean, axis=1)
    log_likelihood = -0.5 * exponent_term - 0.5 * np.log(det_cov+0.00000000000000001) - 0.5 * len(mean) * np.log(2 * np.pi)
    return log_likelihood


# Classify samples in the test set
predicted_classes = []
for sample in x_test:
    probabilities = []
    for i in range(10):
        probability = qda(sample.flatten(), arr_mean[i], arr_cov[i])
        probabilities.append(probability)
    predicted_class = np.argmax(probabilities)
    predicted_classes.append(predicted_class)

# Evaluate accuracy
total_samples = len(y_test)
correct_predictions = np.sum(predicted_classes == y_test)
accuracy = correct_predictions / total_samples
print("Overall Accuracy:", accuracy)

# Calculate class-wise accuracy
class_accuracy = []
for i in range(10):
    class_indices = np.where(y_test == i)[0]
    correct_predictions = np.sum(predicted_classes[class_indices] == y_test[class_indices])
    class_accuracy.append(correct_predictions / len(class_indices))

print("Class-wise Accuracy:", class_accuracy)
