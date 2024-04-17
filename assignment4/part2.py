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
dx0=np.where(y_train == -1)[0]
dx1=np.where(y_train == 1)[0]
x_val=np.concatenate([x_train[dx0[:1000]],x_train[dx1[:1000]]],axis=0)
y_val=np.concatenate([y_train[dx0[:1000]],y_train[dx1[:1000]]],axis=0)
x_train_new=np.concatenate([x_train[dx0[1000:]],x_train[dx1[1000:]]],axis=0)
y_train_new=np.concatenate([y_train[dx0[1000:]],y_train[dx1[1000:]]],axis=0)
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
class decstump:
    def __init__(self):
        self.threshold = None
        self.alpha = None
        self.fidx = None
    def split(self,x,y):
        nsamps,nfeats=x.shape
        minssr=float('inf')
        bfeat=None
        bthresh=None
        for ithfeat in range(nfeats):
            x_column=x[:,ithfeat]
            thresholds=np.unique(x_column)
            thresholds.sort()
            split_points=[(thresholds[i]+thresholds[i-1])/2 for i in range(1,len(thresholds))]
            for threshold in split_points:
                left=y[x_column<threshold]
                right=y[x_column>=threshold]
                ssr=np.sum((left-np.mean(left))**2)+np.sum((right-np.mean(right))**2)
                if ssr<minssr:
                    minssr=ssr
                    bfeat=ithfeat
                    bthresh=threshold
        self.fidx=bfeat
        self.threshold=bthresh
        return minssr
    def predict(self,x):
        nsamps=x.shape[0]
        xcolum=x[:,self.fidx]
        predictions=np.zeros(nsamps,dtype=float)
        predictions[xcolum<self.threshold]=-0.015
        predictions[xcolum>=self.threshold]=0.015
        return predictions
numiter=300
trees=[]
residuals=y_train_new.copy().astype(float)
for t in range(numiter):
    stump=decstump()
    ssr=stump.split(x_train_new,residuals)
    stump.alpha=1
    trees.append(stump)
    residuals-=stump.alpha*stump.predict(x_train_new)
    valpred=np.sum([tree.alpha*tree.predict(x_val) for tree in trees],axis=0)
    valmse=np.mean((y_val-valpred)**2)
    print(f"Iteration {t}, Validation MSE: {valmse}")
test_pred=np.sum([tree.alpha*tree.predict(x_test) for tree in trees],axis=0)
test_mse = np.mean((y_test-test_pred)**2)
print(f"Test MSE: {test_mse}")
plt.figure(figsize=(10, 6))
valpreds = [np.sum([tree.alpha * tree.predict(x_val) for tree in trees[:i+1]],axis=0) for i in range(numiter)]
plt.plot(range(1,numiter + 1),[np.mean((y_val-pred)**2) for pred in valpreds])
plt.xlabel("No. of Trees")
plt.ylabel("Val MSE")
plt.show()

