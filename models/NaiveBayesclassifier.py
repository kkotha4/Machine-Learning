import pandas as  pd
import math
import random
import numpy as np

class NaiveBayesclassifier:

    def __init__(self):
        print("naive bayes classifier by kashish kothari")


    def split_ratio(self,split_percent,dataframe):

            training_percent=(split_percent/100)
            dataframe_len=len(dataframe)
            index=int(round(dataframe_len*training_percent))

            dataframe=dataframe.sample(frac=1).reset_index(drop=True)
            training_data=dataframe.iloc[:index]
            test_data=dataframe.iloc[index:]
            return training_data, test_data

    def fit_naive_bayes(self,training_data):
            self.summary={}
            self.unique_labels=list(np.unique(training_data["class"]))

            for i in self.unique_labels:
                globals()["data"+str(i)]=training_data[training_data["class"].isin([i])].reset_index(drop=True).drop('class', axis=1)
                standard_deviation=list(globals()["data"+str(i)].std(axis=0))
                mean=list(globals()["data"+str(i)].mean(axis=0))
                final_list=[]
                for j in range(len(mean)):
                    list_main = (mean[j],standard_deviation[j])
                    final_list.append(list_main)

                self.summary[i]=final_list


    def probability_calculation(self,value, mean ,sd):
            exp=np.exp(-1*(np.power((value-mean),2)/(2*np.power(sd,2))))
            prob=(exp/(np.sqrt(2*(np.pi)*(np.power(sd,2)))))
            return prob

    def class_wise_calculation(self,input_row):
            probability={}
            for key,value in self.summary.items():
                prob=1

                for i,j in zip(input_row,value):
                    prob=prob*self.probability_calculation(i,j[0],j[1])

                probability[key]=prob
            return probability

    def predict(self,testing_data ):
            testing_data=testing_data.drop("class",axis=1)
            self.predicted_class=[]
            for i in range(len(testing_data)):
                row=list(testing_data.iloc[i])
                probability=self.class_wise_calculation(row)
                inverse = [(value, key) for key, value in probability.items()]
                value=max(inverse)[1]
                self.predicted_class.append(value)


    def get_accuracy(self,class_label):
        correct=0
        for i,j in zip(class_label,self.predicted_class):
            if (i==j):
                correct+=1
        percentage=(correct/len(class_label))*100
        return percentage




if __name__ == "__main__" :
        print("my first ml algorithm")
        naivebayes=NaiveBayesclassifier()
        dataframe=pd.read_csv("pima-indians-diabetes.csv",header=None,names=["a","b","c","d","e","f","g","h","class"])
        training_data ,test_data = naivebayes.split_ratio(split_percent=80,dataframe=dataframe)
        print("length of training is {} and length of testing is {}".format(len(training_data),len(test_data)))
        fit=naivebayes.fit_naive_bayes(training_data)
        predict=naivebayes.predict(test_data)
        accuracy=naivebayes.get_accuracy(list(test_data["class"]))
        print("the accuracy of this dataset is {} percent".format(accuracy))
