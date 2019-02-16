import numpy as np
import scipy

class Perceptron():
    def __init__(self):
        """
        Initialises Perceptron classifier with initializing
        weights, alpha(learning rate) and number of epochs.
        """
        self.w = None
        self.alpha = 0.3
        self.epochs = 300

    def train(self, X_train, y_train):
        """
        Train the Perceptron classifier. Use the perceptron update rule
        as introduced in Lecture 3.

        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """

        '''initialising weight matrix'''
        nrows=len(set(y_train))
        ncols=X_train.shape[1]
        self.weight_matrix=np.random.rand(nrows,ncols)
        n=0.3
        for epoch in range(self.epochs):
            value=X_train.dot(self.weight_matrix.transpose())
            for i in range(len(X_train)):

                class_v=np.argmax(value[i])
                if class_v != y_train[i] :
                    self.weight_matrix[class_v]=self.weight_matrix[class_v]-(self.alpha*X_train[i])
                    self.weight_matrix[y_train[i]]=self.weight_matrix[y_train[i]]+(self.alpha*X_train[i])
            '''adding the decay'''
            if epoch == (n*self.epochs):
                self.alpha=self.alpha-0.1
                n=n+0.3





    def predict(self, X_test):
        """
        Predict labels for test data using the trained weights.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        pred=[]
        for i in range(len(X_test)):
           value=X_test[i].dot(self.weight_matrix.transpose())
           class_v=np.argmax(value)
           pred.append(class_v)

        return pred 
