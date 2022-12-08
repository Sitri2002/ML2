import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd

def process_data(file):
    df = pd.read_csv(file)
    df = df[df['Age'].notna()]
    #  change male/female to 0/1
    df.loc[ df['Sex'] == 'male', 'Sex'] = 0
    df.loc[ df['Sex'] == 'female', 'Sex'] = 1
    return df

def one_hot(target):
    return np.where(target == 1, [0,1], [1,0])

df = process_data('docs/train_data_titanic.csv')

val_df = df.iloc[len(df.index)-101:,:]
train_df = df.iloc[:len(df.index)-101,:]

A0 = np.array([train_df['Age'], train_df['Sex'], train_df['Sib']+ train_df['Parch'],
 train_df['Pclass']]).T.astype(float)

test_input = np.array([val_df['Age'], val_df['Sex'], val_df['Sib']+val_df['Parch'],
 val_df['Pclass']]).T.astype(float)

Y = np.array([train_df['Survived']]).T.astype(int)
test_output = np.array([val_df['Survived']]).T.astype(int)
test_output = one_hot(test_output)
m , n = A0.shape



def init_params():
    xavier = np.sqrt(3*1/((5+2)/2))
    he = np.sqrt(3*2/4)
    W1 = np.random.uniform(-he,he,[4,5])
    W2 = np.random.uniform(-xavier,xavier,[5,2])
    b1 = np.random.uniform(-he,he,[1,5])
    b2 = np.random.uniform(-xavier,xavier,[1,2])
    return W1, W2, b1, b2

def ReLU(X):
    return np.maximum(0,X)

def dReLU(X):
    return np.where(X > 0, 1, 0)

def softmax(X):
    return np.exp(X)/np.sum(np.exp(X), axis = 1, keepdims = True)

def loss(X, Y):
    return -np.sum(Y*np.log2(X),axis = 0, keepdims = True)

def forward_prop(A0, W1, W2, b1, b2):
    Z1 = A0.dot(W1) + b1
    A1 = ReLU(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backprop(A1, A2, W2, A0):
    dZ2 = A2 - one_hot(Y)
    dW2 = 1/m*A1.T.dot(dZ2)
    db2 = 1/m*np.sum(dZ2, axis = 0)
    dZ1 = dZ2.dot(W2.T)*dReLU(A1)
    dW1 = 1/m*A0.T.dot(dZ1)
    db1 = 1/m*np.sum(dZ1, axis = 0)
    return dW2, db2, dW1, db1

def predict(input, W1, W2, b1, b2):
    A2 = forward_prop(input, W1, W2, b1, b2)[3]
    return A2
L_list = []
Acc_list = []
def gradient_descent(epochs, alpha):
    W1, W2, b1, b2 = init_params()
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_prop(A0, W1, W2, b1, b2)
        dW2, db2, dW1, db1 = backprop(A1, A2, W2, A0)
        W1 -= alpha*dW1
        b1 -= alpha*db1
        W2  -= alpha*dW2
        b2 -= alpha*db2
        accurate_guess = 0
        #print(f'Iteration:{i}' )
        
        L = 0

        if i % 10 == 0:
            #print(f'{W1}\n {W2}\n {b1}\n {b2}\n')
            for x in range(test_input.shape[0]):
                ind = np.random.randint(0, test_input.shape[0])
                pred = predict(test_input[ind], W1, W2, b1, b2)
                pred = pred.flatten()
                s = test_output[ind]
                if (pred[1] > 0.5 and s[1] == 1) or pred[1] < 0.5 and s[1] == 0:
                    accurate_guess +=1
                L += loss(pred, s)
            Acc_list.append(accurate_guess/test_input.shape[0]*100)
            L_list.append(L/test_input.shape[0])
            print(f'Loss: {L/test_input.shape[0]}')
            print(f'Accuracy: {accurate_guess/test_input.shape[0]*100}%')
    plt.figure(1)
    plt.plot(Acc_list)
    plt.figure(2)
    plt.plot(L_list)
    

gradient_descent(500, 0.03) 
plt.show()

