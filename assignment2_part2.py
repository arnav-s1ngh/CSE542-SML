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


for i in range(len(x_test)):
    x_test[i] = x_test[i].flatten()
x_test=np.array(x_test)
