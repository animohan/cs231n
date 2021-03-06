import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    #print("Num_test = ", num_test) 
    #print("Num_train = ", num_train)
    #print("Train Shape = ", self.X_train.shape)
    #print("Test Shape = ", X.shape)
    
    dists = np.zeros((num_test, num_train))
    #print("xtrain", self.X_train[0,0:10])
    #print("xtest", X[0,0:10])
    #print("xtest sum",np.sum(X[0,:]))
    #print("0th entry", np.sum((self.X_train[0]- X[0])**2))
    #print("10 Differences", (self.X_train[0]- X[0])[0:10])
    for i in np.arange(num_test):
      for j in np.arange(num_train):
        dists[i,j] = (np.sum((self.X_train[j,:] - X[i,:])**2))
        
        ########################COMPLETED ABOVE##############################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in np.arange(num_test):
        dists[i,:] = (np.sum((self.X_train - X[i])**2,axis = 1))
 
      ####################COMPLETED##########################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      #pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    sqr_Xtrain = np.sum(self.X_train**2, axis = 1)
    sqr_Xtest = np.sum(X**2, axis = 1)
    sqr_Xtrain = np.reshape(sqr_Xtrain, [1,sqr_Xtrain.shape[0]])
    sqr_Xtest = np.reshape(sqr_Xtest, [sqr_Xtest.shape[0],1])
    dists = sqr_Xtrain + sqr_Xtest - 2*X.dot(self.X_train.T)
    #print("Reshaped Xtrain:", sqr_Xtrain.shape)
    #print("Reshaped Xtest:", sqr_Xtest.shape)
    #dists = np.sum((reshaped_Xtrain - reshaped_Xtest)**2, axis = 2)
                                   
    #####################COMPLETED###########################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    #print("In Predict labels: Distance shape", dists.shape)
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in np.arange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      
      y_pred = np.ones(num_test)*np.NaN
      nearest_neighbor_index = np.empty([k])  
      for i in np.arange(num_test):
             nearest_neighbor_index= np.argsort(dists[i,:])[0:k]
             #print(nearest_neighbor_index)
             #print(self.y_train.shape)
             nearest_neighbor = self.y_train[nearest_neighbor_index]
             
             y_pred[i] = np.argmax(np.bincount(nearest_neighbor))
                # np.bincount creates a counter from 0 to max number in passed value.
                # and gives a count of each number e.g np.bincount([0,2,2,3]) = [1,0,2,1]
                # Argmax gives the index of the maximum count which for bincount is the 
                # number that has the highest count
    return y_pred                                
           
      ####################COMPLETED############################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      #pass
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #pass
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################



