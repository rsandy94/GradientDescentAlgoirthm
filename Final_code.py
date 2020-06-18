import pandas
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import seed
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,roc_curve
from scipy import stats

df = pandas.read_csv("D:/UTD/3rseem/AML/Assignment1/energydata_complete.csv")
df=df.drop(['date'],axis=1)
print df['Appliances'].describe()

pandas.set_option('display.max_columns', None)

#Corrplot
cormat=df.corr()
print(cormat)
cormat.to_csv("D:/UTD/3rseem/AML/Assignment1/cormat.csv",index=False)
k=27
cols = cormat.nlargest(k, 'Appliances')['Appliances'].index
cm = np.corrcoef(df[cols].values.T)
print cm[0]
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, ax=ax, cmap="YlGnBu",
            linewidths=0.1, yticklabels=cols.values,
            xticklabels=cols.values)

plt.show()
seed(1)
split= np.random.rand(len(df))<0.7
train=df[split]
test=df[~split]
cols=['lights','T1','RH_1','T2','RH_2','T3','RH_3','T4','T6','RH_6','RH_7','T8','RH_8','RH_9','T_out','RH_out','Windspeed']
Xtrain=train[cols]
std_Xtrain = preprocessing.scale(Xtrain)
Ytrain=train[['Appliances']]
Ytrain = Ytrain.values.reshape(Ytrain.shape[0])
Xtest=test[cols]
std_Xtest = preprocessing.scale(Xtest)
Ytest=test[['Appliances']]
Ytest = Ytest.values.reshape(Ytest.shape[0])


def linear_regression(x,y,alpha,iterations,threshold):
    n = x.shape[1]
    inter_col=np.ones((x.shape[0],1))
    x=np.concatenate((inter_col,x),axis=1)
    thetha=np.zeros(n+1)
    p= predictions(thetha,x,n)
    thetha,cost=GradientDescent(thetha,alpha,iterations,p,x,y,n,threshold)
    return thetha,cost

def linear_regressiontest(xtrain,ytrain,xtest,ytest,alpha,iterations,threshold):
    n = xtrain.shape[1]
    print(xtrain.shape[0])
    inter_col1=np.ones((xtrain.shape[0],1))
    inter_col2 = np.ones((xtest.shape[0], 1))
    xtrain=np.concatenate((inter_col1,xtrain),axis=1)
    xtest = np.concatenate((inter_col2, xtest), axis=1)
    thetha=np.zeros(n+1)
    p= predictions(thetha,xtrain,n)
    thetha,cost1,cost2,index=GradientDescenttest(thetha,alpha,iterations,p,xtrain,ytrain,xtest,ytest,n,threshold)
    return thetha,cost1,cost2,index



def predictions(thetha,x,n):
    p=np.ones((x.shape[0],1))
    thetha=thetha.reshape(1,n+1)
    for i in range(0,x.shape[0]):
        p[i]=float(np.matmul(thetha,x[i]))
    p=p.reshape(x.shape[0])
    return p


def GradientDescent(thetha,alpha,iterations,p,x,y,n,threshold):
    cost=np.ones(iterations,dtype=np.float64)
    rows=x.shape[0]
    index=0
    for i in range(0,iterations):
        thetha[0]=thetha[0]-(alpha/rows)*sum(p-y)
        for j in range(1,n+1):
            thetha[j]=thetha[j]-(alpha/rows)*sum((p-y)*x.transpose()[j])
        p=predictions(thetha,x,n)
        sq_errors = np.square(p-y)
        sum_sq_errors=sum(sq_errors)
        cst= sum_sq_errors*0.5*(1/float(rows))
        cost[i]=cst
        if (i!=0 and cost[i-1]-cost[i]<threshold) or (i==iterations-1):
            index=i
            break

    thetha=thetha.reshape(1,n+1)
    return thetha,cost[index]

def GradientDescenttest(thetha,alpha,iterations,p,xtrain,ytrain,xtest,ytest,n,threshold=0):
    cost1=np.ones(iterations,dtype=np.float64)
    cost2 = np.ones(iterations, dtype=np.float64)
    index = 0
    p1 = np.ones((xtest.shape[0], 1))
    rows=xtrain.shape[0]
    rows1 = xtest.shape[0]
    for i in range(0,iterations):
        thetha[0]=thetha[0]-(alpha/rows)*sum(p-ytrain)
        for j in range(1,n+1):
            thetha[j]=thetha[j]-(alpha/rows)*sum((p-ytrain)*xtrain.transpose()[j])
        p=predictions(thetha,xtrain,n)
        sq_errors1 = np.square(p-ytrain)
        sum_sq_errors1=sum(sq_errors1)
        cost1[i]= sum_sq_errors1*0.5*(1/float(rows))

        p1 = predictions(thetha, xtest, n)
        sq_errors2 = np.square(p1 - ytest)
        sum_sq_errors2 = sum(sq_errors2)
        cost2[i] = sum_sq_errors2 * 0.5 * (1 / float(rows1))
        if (i!=0 and cost2[i-1]-cost2[i]<threshold) or (i==iterations-1):
            index=i
            break
    thetha=thetha.reshape(1,n+1)
    return thetha,cost1,cost2,index


if __name__=='__main__':


# Experiment 1 alpha=0.1
    theta1, cost1, cost2, index = linear_regressiontest(std_Xtrain, Ytrain, std_Xtest, Ytest, 0.1, 1000, 0)
    print cost1[-1]
    print cost2[-1]
    n_iterations = [x for x in range(1,1001)]
    plt.plot(n_iterations,cost1,cost2)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.legend(['Training data','Test data'])
    plt.show()
# Experiment 1 alpha=0.01
    theta1, cost1, cost2, index = linear_regressiontest(std_Xtrain, Ytrain, std_Xtest, Ytest, 0.01, 1000, 0)
    print cost1[-1]
    print cost2[-1]
    n_iterations = [x for x in range(1,1001)]
    plt.plot(n_iterations,cost1,cost2)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.legend(['Training data','Test data'])
    plt.show()
# Experiment 1 alpha=0.001
    theta1, cost1, cost2, index = linear_regressiontest(std_Xtrain, Ytrain, std_Xtest, Ytest, 0.001, 5000, 0)
    print cost1[-1]
    print cost2[-1]
    n_iterations = [x for x in range(1,5001)]
    plt.plot(n_iterations,cost1,cost2)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.legend(['Training data','Test data'])
    plt.show()
# Experiment 1 alpha=0.0001
    theta1, cost1, cost2, index = linear_regressiontest(std_Xtrain, Ytrain, std_Xtest, Ytest, 0.0001, 10000, 0)
    print cost1[-1]
    print cost2[-1]
    n_iterations = [x for x in range(1,10001)]
    plt.plot(n_iterations,cost1,cost2)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.legend(['Training data','Test data'])
    plt.show()

#Experiment 2
     #Threshold calculation -Train data
    dict1={}
    threshold=[10,5,1,0.1,0.01,0.001,0.0001]

    for th in threshold:

        theta1,cost1=linear_regression(std_Xtrain,Ytrain,0.1,1000,th)
        dict1[th]=cost1
    print dict1

    #Threshold calculation -Test data
    dict2={}

    for th in threshold:

        theta1,cost1=linear_regressiontest(std_Xtrain,Ytrain,std_Xtest,Ytest,0.1,1000,th)
        dict2[th]=cost1
    print dict2


    plt.plot(*zip(*sorted(dict1.items())))
    for x, y in _dict1.items():
        label = "({0},{1:.4f})".format(x,y)

        plt.annotate(label,
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')
    plt.xlabel('Threshold Value')
    plt.ylabel('Cost error')
    plt.title("Test Data")
    plt.show()

    plt.plot(*zip(*sorted(dict2.items())))
    for x, y in dict2.items():
        label = "({0},{1:.4f})".format(x,y)

        plt.annotate(label,
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')
    plt.xlabel('Threshold Value')
    plt.ylabel('Cost error')
    plt.title("Test Data")
    plt.show()
#Regression with alpha=0.1 and threshold =0.01

    theta1, cost1, cost2, index = linear_regressiontest(std_Xtrain, Ytrain, std_Xtest, Ytest, 0.1, 1000, 0.01)
    print(theta1)
    print cost1[-1]
    print cost2[-1]



#Experiment 3
#Random varible selection
    ran_cols=random.sample(cols,10)
    Xtrain=train[ran_cols]
    std_Xtrain = preprocessing.scale(Xtrain)
    Xtest=test[cols]
    # Xtrain=train.drop(['Appliances'],axis=1)
    std_Xtest = preprocessing.scale(Xtest)
    theta1,cost1,cost2,index=linear_regressiontest(std_Xtrain,Ytrain,std_Xtest,Ytest,0.1,1000,0)
    print(theta1)
    print cost1[-1]
    print cost2[-1]

#Experiment 4
# Choosing 10 variables on my won
    cols=['T1','RH_1','T2','T3','RH_3','T4','T5','T7','RH_7','T_out']
    Xtrain=train[ran_cols]
    std_Xtrain = preprocessing.scale(Xtrain)
    Xtest=test[cols]
    # Xtrain=train.drop(['Appliances'],axis=1)
    std_Xtest = preprocessing.scale(Xtest)
    theta1,cost1,cost2,index=linear_regressiontest(std_Xtrain,Ytrain,std_Xtest,Ytest,0.1,1000,0)
    print(theta1)
    print cost1[-1]
    print cost2[-1]




 # Logistic regression

    df['Efficiency_binary'] = np.where(df['Appliances'] >= 60, 1, 0)
    x_log= df.drop(['Efficiency_binary'],axis=1)
    y_log=df['Efficiency_binary']
    x_train, x_test, y_train, y_test = train_test_split(x_log, y_log,train_size=0.7, random_state=1)
    std_Xtrain = preprocessing.scale(x_train)
    std_Xtest = preprocessing.scale(x_test)

    lr = LogisticRegression()
    lr.fit(std_Xtrain, y_train)
    print lr.score(std_Xtrain, y_train)
    THRESHOLD = 0.05
    probs=lr.predict_proba(std_Xtest)

    preds = np.where(lr.predict_proba(std_Xtest)[:, 1] > THRESHOLD, 1, 0)
    # y_pred = lr.predict(std_Xtest)
    conf=confusion_matrix(y_test, preds)
    print conf
    print(classification_report(y_test, preds))

    fpr, tpr, _ = roc_curve (preds, y_test, drop_intermediate=False)

    plt.figure()
    ##Adding the ROC
    plt.plot(fpr, tpr, color='red',
             lw=2, label='ROC curve')
    ##Random FPR and TPR
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    ##Title and label
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC curve')
    plt.show()


