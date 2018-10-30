
import numpy as np
import matplotlib.pyplot as plt

def load_data(fname):
    x_data = []
    y_data = []
    with open(fname,"r") as f:
        line = f.readline()
        while line:
            data = line.strip().split('\t')
            x_data.append(data[0].split(" "))
            y_data.append(data[1])
            line = f.readline()        
    
    return np.array(x_data).astype(np.float64),np.array(y_data).astype(np.float64)

def sign(x):
    if(x.shape == ()):
        return np.array([1] if (x>0) else [-1])
    ans = []
    for i in range(x.shape[0]):
        ans.append([1] if (x[i]>0) else [-1])
    return np.array(ans)

class perceptron:
    def __init__(self,length):
        self.W = np.zeros((length+1))

    def train(self,x_data,y_data,rate = 1):
        length = x_data.shape[0]
        X_data = np.hstack([np.ones((length,1)),x_data])
        index = [i for i in range(length)]
        np.random.shuffle(index)
        
        count = 0
        update = 0
        j = 0
        
        while(count<length):
            while(j<length):
                y_pre = sign(X_data[index[j]].dot(self.W))
                if(y_pre[0] == y_data[index[j]]):
                    count += 1
                else:
                    self.W += y_data[index[j]]*X_data[index[j]]*rate
                    update += 1
                    count = 0
                j += 1
            j %= length
            
        return update
    
    def train_many(self,x_data,y_data,iterations,rate = 1):
        update_num = []
        length = x_data.shape[1]
        for i in range(iterations):
            self.W = np.zeros((length+1))
            np.random.seed(i)
            update_num.append(self.train(x_data,y_data,rate))
        return update_num
    
    def test(self,x_test,y_test):
        length = x_test.shape[0]
        X_test = np.hstack([np.ones((length,1)),x_test])
        y_pre = sign(X_test.dot(self.W))
        correct = np.sum(y_pre.reshape((-1,)) == y_test.reshape((-1,)))
        
        return correct/length

def main():
    x_data,y_data = load_data("hw1_7_train.txt")

    p = perceptron(x_data.shape[1])
    p.train(x_data,y_data)

    update_num = p.train_many(x_data,y_data,1126)
    print("The average number of update before the algorithm halts is",sum(update_num)/1126)
    
    plt.title('Histogram')
    plt.xlabel("The number of updates")
    plt.ylabel("The frequency of the number")
    plt.hist(update_num,bins = 100)
    plt.savefig("Histogram.jpg")

if __name__ == '__main__':
    main()






