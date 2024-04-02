import numpy as np
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
   def gini(self, y):
       worthless,counts=np.unique(y, return_counts=True)
       mean_cnt=counts/len(y)
       return 1-np.sum(mean_cnt**2)
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
   def fit(self,X,y):
       self.tree=self.split(X,y)
   def predict(self,X):
       preds=np.zeros(len(X))
       for i,x in enumerate(X):
           node=self.tree
           while node[1] is None:
               dex,split_val,ltree,rtree=node[0]
               if x[dex]<split_val:
                   node=ltree
               else:
                   node=rtree
           preds[i]=node[1]
       return preds
tree=dectree(mdepth=2)
tree.fit(x_train,y_train)
# finding the classwise accuracy using class list attribution
preds=tree.predict(x_test)
accuracy=np.mean(preds==y_test)
classaccuracy=[np.mean((preds==cl)&(y_test==cl)) for cl in [0,1,2]]
print(f"Overall accuracy is - {accuracy}")
print(f"Classwise accuracy is- {classaccuracy}")

# from collections import Counter
# num_trees = 5
# preds = np.zeros((len(x_test), num_trees), dtype=int)
#
# for i in range(num_trees):
#    dex = np.random.choice(len(x_train), len(x_train), replace=True)
#    X_bag = x_train[dex]
#    y_bag = y_train[dex]
#    tree = dectree(mdepth=3)
#    tree.fit(X_bag, y_bag)
#    preds[:, i] = tree.predict(x_test)
#
# bagged_preds = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=1, arr=preds)
# accuracy = np.mean(bagged_preds == y_test)
# classaccuracy = [np.mean((bagged_preds == c) & (y_test == c)) for c in [0, 1, 2]]
# print(f"\nBagging overall accuracy: {accuracy:.3f}")
# print(f"Bagging class-wise accuracy: {classaccuracy}")


