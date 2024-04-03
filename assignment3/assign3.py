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
for i in range(len(x_train)):
    x_train[i]=x_train[i].flatten()
x_train=np.array(x_train)
#selcting classes-0,1,2
for i in range(len(x_test)):
    x_test[i]=x_test[i].flatten()
x_test=np.array(x_test)
y_test=np.array(y_test)
y_train=np.array(y_train)
x_train=x_train[np.isin(y_train,[0,1,2])]
y_train=y_train[np.isin(y_train,[0,1,2])]
x_test=x_test[np.isin(y_test,[0,1,2])]
y_test=y_test[np.isin(y_test,[0,1,2])]
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
x_train=pca(x_train,10)
x_test=pca(x_test,10)
print(x_train.shape)
print(x_test.shape)
#creating a decision tree
class dectree:
   def __init__(self,mdepth=None):
       self.mdepth=mdepth
       self.tree=None
    #using gini for measuring impurity
   def gini(self, y):
       worthless,counts=np.unique(y, return_counts=True)
       mean_cnt=counts/len(y)
       return 1-np.sum(mean_cnt**2)
   #chooses pnt which has least impurity acc. gini
   def split(self,x,y,depth=0):
       if len(np.unique(y))==1 or (self.mdepth is not None and depth>=self.mdepth):
           return None,np.bincount(y).argmax()
       best_gini=np.inf
       supsplit=None
       bestbranch=None
       for dex in range(x.shape[1]):
           sortedbranch=np.argsort(x[:,dex])
           xsorted=x[sortedbranch]
           ysorted=y[sortedbranch]
           for i in range(1,len(x)):
               if xsorted[i,dex]!=xsorted[i-1,dex]:
                   left_y=ysorted[:i]
                   right_y=ysorted[i:]
                   gini=(len(left_y)/len(y))*self.gini(left_y)+(len(right_y)/len(y))*self.gini(right_y)
                   if gini<best_gini:
                       best_gini=gini
                       supsplit=(dex,xsorted[i, dex])
                       bestbranch=dex
       if supsplit is None:
           return None,np.bincount(y).argmax()
       leftbranch=x[:,bestbranch]<supsplit[1]
       rightbranch=x[:,bestbranch]>=supsplit[1]
       ltree=self.split(x[leftbranch],y[leftbranch],depth+1)
       rtree=self.split(x[rightbranch],y[rightbranch],depth+1)
       return (supsplit[0],supsplit[1],ltree,rtree),None
   def fit(self,x,y):
       self.tree=self.split(x,y)
    #predicts labels for x_test
   def predict(self,x):
       preds=np.zeros(len(x))
       for i,x in enumerate(x):
           node=self.tree
           while node[1] is None:
               dex,split_val,ltree,rtree=node[0]
               if x[dex]<split_val:
                   node=ltree
               else:
                   node=rtree
           preds[i]=node[1]
       return preds
def class_wise_accuracy(preds,y_true):
    accuracies=[]
    for c in np.unique(y_true):
        mask=(y_true== c)
        class_preds=preds[mask]
        class_true=y_true[mask]
        accuracy=(np.mean(class_preds== class_true))
        accuracies.append(accuracy)
    return accuracies
def mostcom(x):
    unique,cnt=np.unique(x,return_counts=True)
    maxcnt=np.max(cnt)
    mcind = np.argmax(cnt)
    if maxcnt<3:
        return -1
    else:
        return unique[mcind]
tree1=dectree(mdepth=2)
tree1.fit(x_train,y_train)
# finding the classwise accuracy using class list attribution
preds=tree1.predict(x_test)
accuracy=np.mean(preds==y_test)
# classaccuracy=[np.mean((preds==cl)&(y_test==cl)) for cl in [0,1,2]]
print(f"Overall accuracy is - {accuracy}")
print(f"Classwise accuracy is- {class_wise_accuracy(preds,y_test)}")
preds2=np.zeros((len(x_test),5),dtype=int)
for i in range(5):
   dex=np.random.choice(len(x_train),len(x_train),replace=True)
   bax=x_train[dex]
   bay=y_train[dex]
   tree2=dectree(mdepth=3)
   tree2.fit(bax,bay)
   preds2[:,i]=tree2.predict(x_test)
bpreds=np.apply_along_axis(mostcom,axis=1,arr=preds2)
accuracy=np.mean(bpreds==y_test)
print(f"Bagging overall- {accuracy}")
print(f"Bagging classwise- {class_wise_accuracy(bpreds,y_test)}")
