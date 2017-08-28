import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] #W shape (DxC) = (Dimension of input data point * Number of Classes)
  num_train = X.shape[0] # X shape = (NxD) = (Number of Datapoints * Dimension of Each datapoint)
  loss = 0.0
  for i in np.arange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    for j in np.arange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss = loss + margin
        #dW[:,j] = dW[:,j] + X[j] 
        dW[:,j] = X[j] 
      else:
        #dW[:,j] = dW[:,j] + 0
        dW[:,j] = 0
     
    W = W - dW
    
    '''
         Reasoning for dW
         L = sum( max(0, W*Xj - W*Xi) where i = true class and j = all wrong classes
         dL/dW = 0 if W*Xj - W*Xi < 0
         dL/dW = dL/dW1 (function) + dL/dW2 (function).. i.e it is the sum of partial derivatives
         
         Matrix Structure: W = D*C
         This is for the ith image
         [Xi1 Xi2 ... XiD]      |   |      |
                             *  |   |      |   
                                W1  W2 ... WC
                                |   |      |
                                |   |      |
         Assume true class is 5th
         dL/dW1 = d/dW1( max(0,W1*Xi1 - W5*Xi5) + max(0, W2*Xi2 - W5*Xi5) + ....)
         dL/dW1 = d/dW1 (max(0, W1*Xi1 - W5*Xi5)  # Rest of the partial derivatives are zero
         dL/dW1 = d/dW1(W1*Xi1 - W5*Xi5) if W1*Xi1 - W5*Xi5 > 0
         dL/dW1 = Xi1 - 0 if (W1*Xi1 - W5*Xi5) >0
         Hence
             dL/dW1 = Xi1 if (W1*Xi1 - W5*Xi5) >0
             dL/dW1 = 0  (W1*Xi1 - W5*Xi5)= <0
        
        Additionally:
         We do not change W with each image, but keep on accumulating the gradients in dW
    '''

        
    #Calculate the gradient numerically
    '''
        tempW = W
        gradientLoss = 0.0
        delta = 0.001
        for k in np.arange(W.shape[0]):
                for l in np.arange(W.shape[1]):
                    tempW[k,l] = W[k,l] + delta
                    scores = X[i].dot(tempW)

                    for m in np.arange(num_classes):
                        if m == y[i]:
                            continue
                        margin = scores[m] - correct_class_score + 1
                        if margin > 0:
                            gradientLoss =margin + gradientLoss

                    dW[l,m] = dW[l,m] + (gradientLoss - loss)/delta
    '''
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss = loss/num_train

  # Add regularization to the loss.
  loss = loss + 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
