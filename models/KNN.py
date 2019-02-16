import numpy as np
import scipy

class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        self.k = k

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
        self.x_train=X
        self.y_train=y



    def find_dist(self, X_test):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train.

        Hint : Use scipy.spatial.distance.cdist

        Returns :
        - dist_ : Distances between each test point and training point
        """

        dist_=scipy.spatial.distance.cdist(X_test,self.x_train,metric="euclidean")



        return dist_

    def predict(self, X_test):
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        val=self.find_dist(X_test)

        max_value_index=[]
        predicted_classes=[]
        for i in range(len(val)):

                 ind = np.argpartition(val[i], self.k)[:self.k]

                 max_value_index.append(ind)
        for i in max_value_index:
                         classes=[]
                         for indexes in i:
                              classes.append(y_train[indexes])
                         maximum=Counter(classes)
                         value,count= maximum.most_common()[0]
                         predicted_classes.append(value)




        return predicted_classes
