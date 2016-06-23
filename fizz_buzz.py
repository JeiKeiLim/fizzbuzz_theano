
# coding: utf-8

# # Fizzbuzz for theano Ver 0.9
# Author : Jongkuk Lim (Lim.JeiKei@gmail.com)<br />
# Date : 2016. 06. 23<br />
# Last update : 2016. 06. 23

# ## Import libraries including Multi Layer Perceptron
# You can find MLP script from http://deeplearning.net/tutorial/mlp.html<br />
# However, I modified only its activation function to ReLU which shows promising result in general.<br />

# In[1]:

import numpy as np
import theano
import theano.tensor as T

from mlp import MLP, HiddenLayer


# ## Defining functions
# I had too much fun from this project so ended up, I made functions which really don't need<br />
# But it's fun to look at it anyway

# In[2]:

def binary_decode(b_data):
    result = 0
    for i in range(len(b_data)):
        result += (b_data[i]*(2**i))
    return result.astype('int64')
    
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode_num(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0

def fizz_buzz_encode(i):
    fizz_buzz = fizz_buzz_encode_num(i)
    result = np.array([0, 0, 0, 0])
    result[fizz_buzz] = 1
    return result

def fizz_buzz_encode_str(i):
    fizz_buzz = fizz_buzz_encode_num(i)
    fizz_str = ['.', 'fizz', 'buzz', 'fizzbuzz']
    return fizz_str[fizz_buzz]


# ## Defining experiment constants
# <B>NUM_DIGITS</B> decides how big your number will be and the number of input neuron.<br />
# <B>NUM_HIDDEN</B> will be the number of hidden node for MLP<br />
# <B>batch_size</B> is the number to split training set into batch_size when the data is too big, this will be helpful.<br />
# <B>learning_rate</B> decides whether the classifier will learn fast or not. It's bad idea to put too big number in learning rate however.<br />
# <B>L1</B> and <B>L2</B> regularizer is used for training and regularizing its error cost.<br />
# <B>n_test</B> is the number of testing data. 

# In[3]:

NUM_DIGITS = 10
NUM_HIDDEN = 1000

batch_size = 128
learning_rate = 0.01
L1_reg = 0.00
L2_reg = 0.0001

n_test = int((2**NUM_DIGITS)*0.1)


# ## Making training and testing set

# In[4]:

train_x = np.array([binary_encode(i, NUM_DIGITS)     for i in range(n_test+1, 2 ** NUM_DIGITS)]).astype('float64')
train_y = np.array([fizz_buzz_encode_num(i)          for i in range(n_test+1, 2 ** NUM_DIGITS)]).astype('int32')

test_x = np.array([binary_encode(i, NUM_DIGITS)      for i in range(1, n_test)]).astype('float64')
test_y = np.array([fizz_buzz_encode_num(i)           for i in range(1, n_test)]).astype('int32')


# ## Loading training and testing set onto VRAM

# In[5]:

tr_x = theano.shared(train_x)
tr_y = theano.shared(train_y)
te_x = theano.shared(test_x)
te_y = theano.shared(test_y)

n_train_batches = tr_x.get_value(borrow=True).shape[0] // batch_size

index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')


# ## Defining Models
# 

# In[6]:

rng = np.random.RandomState(1234)
classifier = MLP(
    rng=rng,
    input=x,
    n_in=NUM_DIGITS,
    n_hidden=NUM_HIDDEN,
    n_out=4
)

cost = (classifier.negative_log_likelihood(y) 
       + L1_reg * classifier.L1
       + L2_reg * classifier.L2_sqr)

test_model = theano.function(
    inputs=[],
    outputs=classifier.errors(y),
    givens={
        x:te_x,
        y:te_y
    }
)

gparams = [T.grad(cost, param) for param in classifier.params]
updates = [
    (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)
]

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: tr_x[index * batch_size : (index+1) * batch_size],
        y: tr_y[index * batch_size : (index+1) * batch_size]
    }
)


# ## Training
# Since we do not have validation set which quite often exist in deep learning session, <br />
# Training will stop when either testing error is zero or training improvement is lower than stop error we set.<br />

# In[7]:

def print_status(n_iter, minibatch_avg_cst, test_err):
    print("Iter : ", n_iter, ", Average batch cost : " , minibatch_avg_cst, ", Test Error : ", str(test_err*100), "%", sep="")

best_loss = np.inf
prev_minibatch_avg_cst = np.inf
minibatch_avg_cst = np.inf

validate_freq = 1000
max_epoch = 50000

stop_err = 0.001

for i in range(max_epoch):
    minibatch_avg_cst = []
    
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cst.append( train_model(minibatch_index) )
    
    minibatch_avg_cst = np.mean(minibatch_avg_cst)
    
    if i%validate_freq == 0:
        test_err = test_model()
        
        if test_err < best_loss:
            best_loss = test_err
            
        if prev_minibatch_avg_cst-minibatch_avg_cst < stop_err or test_err == 0.0:
            break
        
        prev_minibatch_avg_cst = minibatch_avg_cst
            
        print_status(i, minibatch_avg_cst, test_err)
        
print_status(i, minibatch_avg_cst, test_model())
print('Your training is over.')


# ## Defining a model to check how the testing result came out
# Test model only shows its error rate.<br />
# However, we need to see what its answer was in order to analyze the model from time to time.

# In[8]:

check_y_model = theano.function(
    inputs=[],
    outputs=classifier.logRegressionLayer.y_pred,
    givens={
        x:te_x
    }
)


# ## Checking which data was failed to classify
# If test error is zero, it will just show every data

# In[9]:

y_from_model = check_y_model()

fizz_str = ['.', 'fizz', 'buzz', 'fizzbuzz']

for i,j,k in zip(test_x, test_y, y_from_model):
    if test_err == 0:
        print(binary_decode(i), " : ", fizz_str[j], ",", fizz_str[k], ', isDiff :', (j!=k), sep="")
    elif j != k:
        print(binary_decode(i), " : ", fizz_str[j], ",", fizz_str[k], ', isDiff :', (j!=k), sep="")
        

