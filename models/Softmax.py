import numpy as np

class Softmax():
    def __init__(self):
        """
        Initialises Softmax classifier with initializing
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 0.2
        self.epochs = 300
        self.reg_const = 0.01

    def calc_gradient(self, X_train, y_train):
        """
        Calculate gradient of the softmax loss

        Inputs have dimension D, there are C classes, and we operate on minibatches
        of N examples.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing a minibatch of data.
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.

        Returns:
        - gradient with respect to weights W; an array of same shape as W
        """
        grad_w=self.w
        weight_vector=X_train.dot(grad_w.T)
        weight_vector=(weight_vector.T-(np.max(weight_vector,axis=1))).T
        exponential_value=np.exp(weight_vector)


        for i in range(len(exponential_value)):

              sum_v=np.sum(exponential_value[i])
              exponential_value[i]=exponential_value[i]/sum_v
              exponential_value[i,y_train[i]]-=1
        weights_batch=exponential_value.T.dot(X_train)

        grad_w=grad_w-weights_batch*(self.alpha*self.reg_const/len(X_train))






        return grad_w

    def train(self, X_train, y_train):
        """
        Train Softmax classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;

        Hint : Operate with Minibatches of the data for SGD
        """
        nrows=len(set(y_train))
        ncols=X_train.shape[1]
        self.w=np.random.random((nrows,ncols))
        batch_size_list=[]
        batchsize=100
        batch_size_list=[[X_train[size:size+batchsize],y_train[size:size+batchsize]]for size in range(0,len(X_train),batchsize)]


        for i in range(self.epochs):
          for batches in batch_size_list:
            self.w=self.calc_gradient(batches[0],batches[1])


    def predict(self, X_test):
        """
        Use the trained weights of softmax classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        pred=[]
        for i in range(len(X_test)):
           value=X_test[i].dot(self.w.transpose())
           class_v=np.argmax(value)
           pred.append(class_v)

        return pred


    
