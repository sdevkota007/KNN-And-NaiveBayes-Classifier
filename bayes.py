from __future__ import division
import operator
import pandas as pd
import numpy as np
import math
import argparse
from ConfusionMatrix import confusionMatrix, plotConfusionMatrix


def readData(file):
    '''
    reads the content of the file and returns X(features) and Y(labels)
    :param file: name of the file
    :return: features, labels
    '''
    dataset = pd.read_csv(file, header=None)
    dataset = dataset.sample(frac=1)

    X = dataset.iloc[:, 0:4].values
    Y = dataset.iloc[:, 4].values
    #return X.reshape((-1,4)), Y.reshape((-1,1))
    return X.reshape((-1,4)), Y

def gaussianPDF(x, mean, stdev):
    '''
    calculates the gaussian probability density
    :param x: sample
    :param mean: mean
    :param stdev: standard deviation
    :return: probability density function
    '''
    var = math.pow(float(stdev), 2)
    lower = math.pow( (2*math.pi * var), 0.5)
    upper = math.exp( -pow( float(x)-float(mean) ,2) / (2*var) )
    return upper/lower

def calculateMeanAndSD(X):
    '''
    calculates mean and standard deviation from X
    :param X: list or array of data
    :return: returns mean and standard deviation
    '''
    mean = np.mean(X)
    std = np.std(X)
    return mean, std

def seggregate(X,Y):
    '''
    takes features and labels as input, and returns a special dictionary object
    :param X: features
    :param Y: labels
    :return: a dictionary object
    eg:  input: X = [[5.1,3.5,1.4,0.2]     Y = [Iris-setosa,
                     [4.9,3.0,1.4,0.2]          Iris-setosa,
                     [...            ]          Iris-versicolor,
                     ...          ..]]          Iris-virginica ]
    returns the features and labels in the form
    {
        label1: {
                    feature1: {
                                data: [5.1, 4.9, .... ],
                                mean: mean value of above data,
                                stdev: stddev of above data,
                    },
                    feature2: {
                                data: [3.5, 3.0, .... ],
                                mean: mean value of above data,
                                stdev: stddev of above data,
                    }
                    feature3: {

                    },
                    feature4: {

                    }
                },
        label2: {

                },
        .
        .
    }
    '''
    num_features = X.shape[1]
    seggregated_data = {}
    for i, data in enumerate(X):
        features = {}
        for j in range(num_features):
            if str(Y[i]) in seggregated_data:
                pass
            else:
                seggregated_data[str(Y[i])] = {}

            if str(j) in seggregated_data[str(Y[i])]:
                temp = np.append(seggregated_data[str(Y[i])][str(j)]["data"], data[j])
                seggregated_data[str(Y[i])][str(j)]["data"] = temp
            else:
                seggregated_data[str(Y[i])][str(j)] = {
                                                        "data" : np.array([ data[j] ]),
                                                        "mean" : None,
                                                        "stdev"  : None
                                                        }
    return fillMeanAndSD(seggregated_data)


def fillMeanAndSD(dictx):
    '''
    calculates mean, standard deviation and inserts into the dictionary object
    :param dictx:
    :return:
    '''
    for cls, cls_value in dictx.items():
        for feature, feature_value in cls_value.items():
            mean, sd = calculateMeanAndSD(feature_value["data"])
            dictx[cls][feature]['mean'] = mean
            dictx[cls][feature]['stdev'] = sd
    return dictx

def selectMaxProbability(classes,probabilities):
    '''
    selects the class with max class probability
    :param classes: list of classes
    :param probabilities: list of probabilities
    :return: returns class with max class probability and max-probability
    '''
    #print(classes, probabilities)
    assert len(classes)==len(probabilities)
    max_prob = max(probabilities)
    max_prob_index = probabilities.index(max_prob)
    best_class = classes[max_prob_index]
    return best_class, max_prob

def getPredictedClass(Xsample, seggregated_class):
    '''
    calculates class probabilites and makes a prediction for the class of the sample
    :param Xsample: features/attributes of a test sample
    :param seggregated_class: dictionary object
    :return: returns the predicted class and probability of sample falling in that class
    '''
    classes = []
    class_probabilities = []
    for cls, cls_value in seggregated_class.items():
        #print(cls)
        classes.append(cls)
        class_probability = 1
        for i, attribute in enumerate(Xsample):
            mean = cls_value[str(i)]['mean']
            stdev = cls_value[str(i)]['stdev']
            probability = gaussianPDF(attribute, mean, stdev)
            #print("Probability: ", probability, " sample: ", attribute, " mean: ", mean, " stddev: ", stdev)
            class_probability = class_probability * probability

        class_probabilities.append(class_probability)

    best_class, best_probability  = selectMaxProbability(classes, class_probabilities)

    return best_class, best_probability


def getAccuracy(true_labels, predictions):
    '''
    calculates accuracy
    :param true_labels:
    :param predictions:
    :return: returns accuracy
    '''
    result = list(map(lambda x,y: (1 if x==y else 0), predictions, true_labels))
    accuracy = sum(result)/(len(result))
    return accuracy


def naiveBayesClassifier(X_train, Y_train, X_test, Y_test):
    '''
    Naive Bayes classifier
    :param X_train: Training set features
    :param Y_train: Training set labels
    :param X_test: Test set features
    :param Y_test: Test set labels
    :return: accuracy, confusion matrix and list of unique classes
    '''
    seggregated_class = seggregate(X_train, Y_train)
    predictions = []
    for i in range(len(X_test)):
        predicted_class, best_probability = getPredictedClass(X_test[i], seggregated_class)
        predictions.append(predicted_class)
        #print(predicted_class, Y_test[i], best_probability)

    accuracy = getAccuracy(Y_test, predictions)
    cm, classes = confusionMatrix(Y_test, predictions)
    return accuracy, cm, classes


def naiveBayesClassifierWithCrossValidation(X, Y, cv=5):
    '''
    Naive Bayes classifier with cross validation
    :param X: Features
    :param Y: Labels
    :param cv: n-fold cross-validation
    :return: average-accuracy, average-confusion matrix and list of unique classes
    '''
    accuracies = []
    confusion_matrices = []
    num_samples = len(X)
    length_test = int((1 / cv) * num_samples)
    u = 0
    v = length_test
    for i in range(cv):
        # print("********************************************************************\n"
        #       "Iteration {0} of {1}-fold cross-validation".format(i + 1, cv))

        # print("u:", u, "v:", v)
        X_test = X[u:v]
        X_train = np.concatenate((X[0:u], X[v:num_samples]), axis=0)
        Y_test = Y[u:v]
        Y_train = np.concatenate((Y[0:u], Y[v:num_samples]), axis=0)
        u = u + length_test
        v = v + length_test
        accuracy, cm, classes = naiveBayesClassifier(X_train, Y_train, X_test, Y_test)

        accuracies.append(accuracy)
        confusion_matrices.append(cm)

        # print("=>Accuracy: ", accuracy)
        # print("=>Confusion Matrix:\n "
        #       "{0}\n"
        #       "{1}\n".format(classes, cm))

    #print(accuracies)
    average_accuracy = sum(accuracies) / len(accuracies)
    sum_cm = 0
    for cm in confusion_matrices:
        sum_cm = sum_cm + cm
    average_cm = sum_cm/len(confusion_matrices)

    return average_accuracy, average_cm, classes


def main():
    #print("Loading Dataset...")
    X, Y = readData("iris.data")
    #print("Completed!")

    accuracy, cm, classes = naiveBayesClassifierWithCrossValidation(X,Y,cv=5)
    print("\n\n***Naive Bayes Classifier (with 5 fold cross validation)***")
    print("\n=>Overall accuracy: ", accuracy)
    print("=>Overall Confusion Matrix:\n "
          "{0}\n"
          "{1}\n".format(classes, cm))
    print("Plotting Confusion Matrix...")
    plotConfusionMatrix(cm, classes,
                        normalize=True,
                        title="Confusion Matrix: Naive Bayes Classifier")

if __name__ == '__main__':
    main()