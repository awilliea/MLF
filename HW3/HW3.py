import numpy as np
import matplotlib.pyplot as plt

# dimension = 20
def load_data(train_filename,test_filename):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    with open(train_filename,'r') as train_file:
        for line in train_file:
            temp = line.strip(' ').split(' ')
            X_train.append(temp[:-1])
            Y_train.append(temp[-1])
    with open(test_filename,'r') as test_file:
        for line in test_file:
            temp = line.strip(' ').split(' ')
            X_test.append(temp[:-1])
            Y_test.append(temp[-1])

    temp_train = np.ones((len(X_train))).reshape((-1,1))
    temp_test = np.ones((len(X_test))).reshape((-1,1))
    
    return np.hstack([temp_train,np.array(X_train,dtype=np.float64)]),np.hstack([temp_test,np.array(X_test,dtype=np.float64)]),np.array(Y_train,dtype=np.float64),np.array(Y_test,dtype=np.float64)


def theta(x):
    return 1/(1+np.exp(-x))

def train(learning_rate,iterations,X_train,Y_train,X_test,Y_test):
    w = np.zeros((X_train.shape[1]))
    Eins = []
    Eouts = []
    for t in range(iterations): 
        temp = np.zeros((w.shape))
        for i in range(X_train.shape[0]):
            temp += theta(-Y_train[i]*(X_train[i].dot(w.T)))*(-Y_train[i]*X_train[i])
        w -= learning_rate*(temp/X_train.shape[0])
        
        Eins.append(get_Error(w,X_train,Y_train))
        Eouts.append(get_Error(w,X_test,Y_test))
    return w,Eins,Eouts

def train_SGD(learning_rate,iterations,X_train,Y_train,X_test,Y_test):
    w = np.zeros((X_train.shape[1]))
    count = 0
    Eins = []
    Eouts = []
    for t in range(iterations): 
        temp = theta(-Y_train[count]*(X_train[count].dot(w.T)))*(-Y_train[count]*X_train[count])
        w -= learning_rate*temp
        count += 1
        if(count == X_train.shape[0]):
            count %= X_train.shape[0]
            
        Eins.append(get_Error(w,X_train,Y_train))
        Eouts.append(get_Error(w,X_test,Y_test))
    return w,Eins,Eouts

def get_Error(w,X,Y):
    predict = np.sign(X.dot(w.T))
    count = 0
    for i in range(X.shape[0]):
        if(predict[i] != Y[i]):
            count += 1
    return count/X.shape[0]


def main():
    X_train,X_test,Y_train,Y_test = load_data('HW3_train.txt','HW3_test.txt')

    w_19_GD,Eins_19_GD,Eouts_19_GD = train(0.01,2000,X_train,Y_train,X_test,Y_test)

    w_20_SGD,Eins_20_SGD,Eouts_20_SGD = train_SGD(0.001,2000,X_train,Y_train,X_test,Y_test)

    x = [i for i in range(2000)]
    plt.figure(figsize=(10,8))
    plt.title('Eins')
    plt.plot(x,Eins_19_GD,label = 'Ein_gradient descent')
    plt.plot(x,Eins_20_SGD,label = 'Ein_SGD')
    plt.legend(loc = 'upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.savefig('Eins.jpg')
    
    print('Save Eins.jpg')
          
    x = [i for i in range(2000)]
    plt.figure(figsize=(10,8))
    plt.title('Eouts')
    plt.plot(x,Eouts_19_GD,label = 'Eout_gradient descent')
    plt.plot(x,Eouts_20_SGD,label = 'Eout_SGD')
    plt.legend(loc = 'upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.savefig('Eouts.jpg')

    print('Save Eouts.jpg')

if __name__ == '__main__':
    main()
