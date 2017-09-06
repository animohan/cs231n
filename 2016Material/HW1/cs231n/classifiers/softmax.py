import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #Shape of W = DxC
  #Shape of X = NxD

  num_train = np.shape(X)[0]
  num_class = np.shape(W)[1]
    
  for i in np.arange(num_train):
    unnormalized_log_prob = X[i].dot(W)
    
    stabilization_factor = np.exp(-1*max(unnormalized_log_prob))
    
    unnormalized_prob = np.exp(unnormalized_log_prob + stabilization_factor)
    
    normalization_factor = np.sum(np.exp(unnormalized_log_prob + stabilization_factor), axis = 0)
    
    #correct_class_score = np.exp(score[y[i]] + stabilization_factor)
    loss = loss + ( -1 * np.log(unnormalized_prob[y[i]]/normalization_factor) )
    
    for j in np.arange(num_class):
        if(j == y[i]):
            dW[:,j] = dW[:,j] + X[i]*(unnormalized_prob[j]/normalization_factor - 1)
        else:
            dW[:,j] = dW[:,j] + X[i]* unnormalized_prob[j]/normalization_factor
     
  loss = loss/num_train
  dW = dW/num_train
  dW = dW + 2*reg*W

  # Add regularization to the loss.
  loss = loss + reg * np.sum(W * W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train = np.shape(X)[0]
  print("Numtrain", num_train)
  num_class = np.shape(W)[1]
  
  stabilization_factor = np.ones([num_train,1])*np.NaN
  correct_class_unnormalized_log_prob = np.ones([num_train,1])*np.NaN  
  
  row_index = np.array([np.arange(num_train)])

  unnormalized_log_prob = X.dot(W)

  stabilization_factor = np.exp(-1*np.max(unnormalized_log_prob, axis = 1))
  
  #Stabilization (for some reason) now has shape of (500,). Forcing it to shape of (500,1)
  stabilization_factor = np.reshape(stabilization_factor,[num_train,1])
  normalization_factor = np.sum(np.exp(unnormalized_log_prob + stabilization_factor), axis = 1)
  normalization_factor = np.reshape(normalization_factor,[num_train,1])
 
  normalized_prob = np.exp(unnormalized_log_prob + stabilization_factor) / normalization_factor

  #Save the Normalized probabilities for the correct class for calculating loss
  correct_class_normalized_log_prob = normalized_prob[row_index, y]
    
  #Subtracting 1 from normalized probability to account for dW for correct class
  normalized_prob[row_index, y] = normalized_prob[row_index, y] - 1

  dW = np.ones([num_train,num_class])*np.NaN   
  dW = X.T.dot(normalized_prob)
  dW = dW / num_train
  dW = dW + 2*reg*W
    
  loss = np.mean( -1 * np.log(correct_class_normalized_log_prob))
  loss = loss + reg * np.sum(W * W)

  ## COMPLETED ################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

