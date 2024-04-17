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
        self.pol=1
        self.threshold=None
        self.alpha=None
        self.fidx=None
    def split(self,x,y,w):
        nsamps,nfeats=x.shape
        minerr=float('inf')
        bfeat=None
        bthresh=None
        for ithfeat in range(nfeats):
            x_column=x[:,ithfeat]
            thresholds=np.unique(x_column)
            thresholds.sort()
            split_points=[(thresholds[i] + thresholds[i-1])/2 for i in range(1,len(thresholds))]
            for threshold in split_points:
                # preds with self.pol==1
                p=1
                predictions=np.ones(nsamps)
                predictions[x_column<threshold]=-1
                # misclassification rate while considering the associated wts
                miffed=w[y!=predictions]
                error=np.sum(miffed)
                if error>0.5:
                    error=1-error
                    p=-1
                if error<minerr:
                    minerr=error
                    bfeat=ithfeat
                    bthresh=threshold
                    self.pol=p
        self.fidx=bfeat
        self.threshold=bthresh
        return minerr
    def predict(self,x):
        nsamps=x.shape[0]
        x_colum=x[:,self.fidx]
        predictions=np.ones(nsamps)
        if self.pol==1:
            predictions[x_colum<self.threshold]=-1
        else:
            predictions[x_colum>=self.threshold]=-1
        return predictions
class adaboost:
    def __init__(self,numofclassifiers=300):
        self.numofclassifiers=numofclassifiers
        self.classifiers=[]
        self.vaccs=[]
    def fit(self,Xtrain,ytrain,Xval,yval):
        nsamps,nfeats=Xtrain.shape
        self.weights=np.ones(nsamps)/nsamps
        for itt in range(self.numofclassifiers):
            classifier=decstump()
            minerr=float('inf')
            for ithfeat in range(nfeats):
                Xcolum=Xtrain[:, ithfeat]
                thresholds=np.unique(Xcolum)
                for threshold in thresholds:
                    # predict with self.pol==1
                    p=1
                    predictions=np.ones(nsamps)
                    predictions[Xcolum<threshold]=-1
                    miffed=self.weights[ytrain!=predictions]
                    error=np.sum(miffed)
                    if error>0.5:
                        error=1-error
                        p=-1
                    # best config is utillised for further calc
                    if error<minerr:
                        classifier.pol=p
                        classifier.threshold=threshold
                        classifier.fidx=ithfeat
                        minerr=error
            classifier.alpha=0.5*np.log((1.0-minerr+(10**-(10)))/(minerr+(10**(-10))))
            # wts updated serially
            predictions=classifier.predict(Xtrain)
            self.weights*=np.exp(-classifier.alpha*ytrain*predictions)
            self.weights/=np.sum(self.weights)
            val_preds=self.predict(Xval)
            val_accuracy=np.mean(val_preds==yval)
            self.vaccs.append(val_accuracy)
            print(f"Iteration No. {len(self.classifiers) + 1}, Val Accuracy = {val_accuracy}")
            self.classifiers.append(classifier)
    def predict(self,x):
        classifier_preds=[classifier.alpha*classifier.predict(x) for classifier in self.classifiers]
        y_pred=np.sum(classifier_preds,axis=0)
        y_pred=np.sign(y_pred)
        return y_pred
    def plotacc(self):
        plt.figure(figsize=(8,6))
        plt.plot(range(1,self.numofclassifiers + 1),self.vaccs)
        plt.xlabel("No. of Trees")
        plt.ylabel("Val Accuracy")
        plt.title("AdaBoost M1 Val Accuracy")
        plt.grid()
        plt.show()
# Usage
model=adaboost(300)
model.fit(x_train_new, y_train_new, x_val, y_val)
model.plotacc()
test_preds=model.predict(x_test)
test_accuracy=np.mean(test_preds==y_test)
print(f"Test Accuracy: {test_accuracy}")
