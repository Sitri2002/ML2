import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import timeit
def process_data(file):
    df = pd.read_csv(file)
    df = df[df['Age'].notna()]
    #  change male/female to 0/1
    df.loc[ df['Sex'] == 'male', 'Sex'] = 0
    df.loc[ df['Sex'] == 'female', 'Sex'] = 1
    return df

def sigmoid(X):
    return 1/(1+np.exp(-X))

def MSE(X, Y):
    return 1/X.shape[0]*np.sum((X-Y)**2)

df = process_data('ML2/train_data_titanic.csv')

val_df = df.iloc[len(df.index)-101:,:]
train_df = df.iloc[:len(df.index)-101,:]

A0 = np.array([train_df['Age'],train_df['Pclass'], train_df['Sex'], train_df['Sib']+ train_df['Parch'],]).astype(float)

test_input = np.array([val_df['Age'], val_df['Pclass'], val_df['Sex'], val_df['Sib']+val_df['Parch'],]).astype(float)

Y = np.array([train_df['Survived']]).astype(int)
test_output = np.array([val_df['Survived']]).astype(int)
m , n = A0.shape

def init_params():
    xavier_init = np.sqrt(6/(n+1))
    W = np.random.uniform(-xavier_init,xavier_init,[4,1])
    b = np.random.uniform(-xavier_init,xavier_init)
    return W, b

def forward_prop(X, W, b):
    Z = W.T.dot(X) + b
    A1 = sigmoid(Z)
    return A1

def back_prop(A1, A0, target):
    dA1 = 2 * (A1 - target)
    dZ1 = dA1 * A1 * (1-A1)
    dW = A0.dot(dZ1.T)/n
    db = np.sum(dZ1)/n
    return dW, db

def update(W, b, dW, db, rate):
    W =  W - dW * rate
    b =b - db * rate
    return W,b
loss = []
accuracy = []
weight1 = []
weight2 = []
weight3 = []
weight4 = []
def gradient_descent(epoch, rate):
    W, b = init_params()
    for i in range(epoch):
        A1 = forward_prop(A0, W, b)
        dW, db = back_prop(A1, A0, Y)
        W, b = update(W, b, dW, db, rate)
        
        accurate_guess = 0
        L = 0
        if i%(epoch/50) == 0:
            for x in range(test_input.shape[1]):
                ind = np.random.randint(0, test_input.shape[1]-1)
                k = test_input[:,ind].reshape(1,4).T
                prediction = sigmoid(W.T.dot(k) + b)[0]
                s = test_output[0][ind].T
                if (prediction > 0.5 and s == 1) or (prediction < 0.5 and s == 0):
                    accurate_guess +=1
                L += (prediction - s) **2
            accuracy.append(accurate_guess/test_input.shape[1])
            loss.append(L/test_input.shape[1])
            print(f'Loss: {L/test_input.shape[1]}')
            print(f'Accuracy: {accurate_guess/test_input.shape[1]*100}%')
            print(W, b)
            weight1.append(W[0].item())
            weight2.append(W[1].item())
            weight3.append(W[2].item())
            weight4.append(W[3].item())

    
    plt.figure(1)
    plt.title(f'Percentage of Acurrate guesses per {epoch} trials')
    plt.plot(accuracy)
   
    plt.figure(2)
    plt.title(f'Mean Squared Error between Prediction and Target after {epoch} trials')
    
    plt.plot(loss)

    plt.figure(3)
    plt.title(f'Weight converge after {epoch} trials')
    plt.plot(weight1, '-b', label = 'Age')
    plt.plot(weight2, '-r', label = 'Status')
    plt.plot(weight3, '-c', label = 'Sex')
    plt.plot(weight4, '-y', label = 'Family')
    plt.legend(loc="upper left")
start = timeit.default_timer()
gradient_descent(40000,0.005)
stop = timeit.default_timer()
print('Time:', stop - start, 's')  
plt.show()
