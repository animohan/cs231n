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
    #print("Iteration:", i)
    #print("\n Score:", scores)
    #print("\n")
    
    margin_greater_than_zero = 0
    for j in np.arange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin >= 0:
        loss = loss + margin
        margin_greater_than_zero = margin_greater_than_zero + 1
        dW[:,j] = dW[:,j] + X[i]
    dW[:,y[i]] = dW[:,y[i]] + (-1 * X[i] * margin_greater_than_zero)
    
    #print("Correct label =",y[i])
    #print("correct gradient row", dW[:,y[i]])
    #print("\n Gradient \n",dW)
    #print("margin greater than zero", margin_greater_than_zero)
  
    
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

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss = loss/num_train
  dW = dW/num_train
  dW = dW + 2*reg*W

  # Add regularization to the loss.
  loss = loss + 0.5 * reg * np.sum(W * W)

  ##########################COMPLETED##########################################
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
  num_train = X.shape[0]
  row_index = np.array([np.arange(num_train)])
  scores = X.dot(W)
  correct_scores = scores[row_index, y]
  scores = scores - correct_scores.T + 1 #updating the scores matrix
        #to see if the score for each class for a particular row of X (X[i])
        #is bigger than correct class score atleast by 1(Delta)
        
  scores[row_index, y ] = 0; #Setting correct label scores to be zero
  loss = np.sum(scores[scores>0]) / num_train
    
   # Add regularization to the loss.
  loss = loss + 0.5 * reg * np.sum(W * W)

  ########################COMPLETED############################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
    
   
  score_bool = scores
  score_bool[score_bool > 0] = 1
  score_bool[score_bool <=0] = 0
  print(score_bool[0:10,0:10])
  
  dW = (X.T).dot(score_bool)
  scaledX = -1 * (X.T) * np.sum(score_bool, axis = 1)
  indexval = np.array(np.arange(num_train))
  dW[:,y[indexval]] = dW[:, y[indexval]] + scaledX[:,indexval]
  dW = dW/num_train
  dW = dW + 2*reg*W
    
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
