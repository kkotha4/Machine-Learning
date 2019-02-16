import numpy as np



class SVM():
    def __init__(self):
        """
        Initialises Softmax classifier with initializing
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 0.01
        self.epochs = 300
        self.reg_const = 0.04

    def calc_gradient(self, X_train, y_train):
        """
          Calculate gradient of the svm hinge loss.

          Inputs have dimension D, there are C classes, and we operate on minibatches
          of N examples.

          Inputs:
          - X_train: A numpy array of shape (N, D) containing a minibatch of data.
          - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means
            that X[i] has label c, where 0 <= c < C.

          Returns:
          - gradient with respect to weights W; an array of same shape as W
         """



        '''for i in range(len(X_train)):

          for classes in set(y_train):
            if classes != y_train[i]:
              calculation=self.w[j].T.dot(X_train[i])-self.w[class_i].T.dot(X_train[i])
              if calculation<1.0:
                self.w[j]=self.w[j]-self.alpha*X_train[i]
                self.w[y_train[i]]=self.w[y_train[i]]+self.alpha*X_train[i]


        self.w=self.w*(1/X_train.shape[0])*self.reg_const'''

        '''#vectorized implementation'''

        '''calculating the dot product of w and X_train'''
        #loss_function=X_train.dot(self.w.T)

        #number=range(0,X_train.shape[0])
        #number=np.array(number)
        #right_class=loss_function[number,y_train]+1

        '''#after getting right class loss value i will subtract from loss function and replace negative value with zero according to formula'''
        #loss_function=loss_function-np.matrix(right_class).T
        #loss_function[loss_function<0]=0

        #loss_function[loss_function>0]=1

        '''#subtracting total no of ones from the '''
        #loss_function[number,y_train]=loss_function[number,y_train]-loss_function.sum(axis=1).T
        '''#multiplying loss function with X_train'''
        #gradient=(self.alpha*X_train).T*loss_function
        #self.w=(gradient.T*self.reg_const)/X_train.shape[0]

        for i in range(len(X_train)):
                value=X_train[i].dot(self.w.transpose())
                class_v=np.argmax(value)
                if class_v != y_train[i] :
                    self.w[class_v]=self.w[class_v]-(self.alpha*X_train[i])
                    self.w[y_train[i]]=self.w[y_train[i]]+(self.alpha*X_train[i])

        self.w=self.w-self.w*((self.alpha*self.reg_const)/len(X_train))


        return self.w

    def train(self, X_train, y_train):
        """
        Train SVM classifier using stochastic gradient descent.

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
            grad_w=self.calc_gradient(batches[0],batches[1])

    def predict(self, X_test):
        """
        Use the trained weights of svm classifier to predict labels for
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
