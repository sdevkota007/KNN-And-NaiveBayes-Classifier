from __future__ import division
import operator
import pandas as pd
import numpy as np
import argparse
from ConfusionMatrix import confusionMatrix, plotConfusionMatrix


def readData(file):
    '''
    reads the content of the file and returns X(features) and Y(labels)
    :param file: name of the file
    :return: features, labels
    '''
    dataset = pd.read_csv(file, header=None)
    data_shuffled = dataset.sample(frac=1)

    X = data_shuffled.iloc[:, 0:4].values
    Y = data_shuffled.iloc[:, 4].values
    #return X.reshape((-1,4)), Y.reshape((-1,1))
    return X.reshape((-1,4)), Y

def euclideanDistance(p1,p2):
    '''
    calculation of euclidean distance
    :param p1:
    :param p2:
    :return:
    '''
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    return dist

def cosineSimilarity(p1,p2):
    '''
    calculation of cosine similarity
    :param p1:
    :param p2:
    :return:
    '''
    cos = np.dot(p1, p2) / (np.sqrt(np.dot(p1, p1)) * np.sqrt(np.dot(p2, p2)))
    return cos


def nearestNeighbors(X_train, Y_train, x_test_sample, k, metric):
    '''
    evaluates K nearest neighbors for the given test sample
    :param X_train: Training set features
    :param Y_train: Training set labels
    :param x_test_sample: features/attributes of a test sample
    :param k: K-nearest neighbors
    :param metric: distance metric, either euclidean or cosine
    :return: list of K nearest neighbors
    '''
    distances = []

    if metric == 'euclidean':
        for i in range(len(X_train)):
            distance = euclideanDistance(X_train[i], x_test_sample)
            distances.append((distance, Y_train[i]))
        distances.sort(key=operator.itemgetter(0))

    elif metric == 'cosine':
        for i in range(len(X_train)):
            distance = cosineSimilarity(X_train[i], x_test_sample)
            distances.append((distance, Y_train[i]))
        distances.sort(key=operator.itemgetter(0), reverse=True)

    else:
        for i in range(len(X_train)):
            distance = euclideanDistance(X_train[i], x_test_sample)
            distances.append((distance, Y_train[i]))
        distances.sort(key=operator.itemgetter(0))

    nearestNeighbors = distances[:k]
    return nearestNeighbors

def getPredictedClass(neighbors):
    '''
    :param neighbors: list of neighbors
    :return: nearest neighbor or predicted class
    '''
    counter = {}
    for item in neighbors:
        class_type = item[1]
        if class_type in counter:
            counter[class_type] +=1
        else:
            counter[class_type] = 1
    nearestNeighbor = max(counter.items(), key = operator.itemgetter(1))[0]
    return nearestNeighbor


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

def knn(X_train, Y_train, X_test, Y_test, k=3, metric="euclidean"):
    '''
    Implementation of K-Nearest Neighbors
    :param X_train: Training set features
    :param Y_train: Training set labels
    :param X_test: Test set features
    :param Y_test: Test set labels
    :param k: K-nearest neighbors
    :param metric: distance metric, either euclidean or cosine
    :return: accuracy, confusion matrix and list of unique classes
    '''
    predictions = []
    for i in range(len(X_test)):
        neighbors = nearestNeighbors(X_train, Y_train, X_test[i], k, metric)
        predicted_class = getPredictedClass(neighbors)
        predictions.append(predicted_class)
        #print(predicted_class, Y_test[i])
    accuracy = getAccuracy(Y_test, predictions)
    cm, classes = confusionMatrix(Y_test, predictions)
    return accuracy, cm, classes

def knnWithCrossValidation(X,Y, k=3, cv=5, metric="euclidean"):
    '''
    KNN with cross validation
    :param X: Features
    :param Y: Labels
    :param k: K-nearest neighbors
    :param cv: n-fold cross-validation
    :param metric: distance metric, either euclidean or cosine
    :return: average-accuracy, average-confusion matrix and list of unique classes
    '''

    accuracies = []
    confusion_matrices = []
    num_samples = len(X)
    length_test = int((1/cv)*num_samples)
    u = 0
    v = length_test
    for i in range(cv):
        print("********************************************************************\n"
              "Iteration {0} of {1}-fold cross-validation".format(i + 1, cv))

        #print("u:", u, "v:", v)
        X_test = X[u:v]
        X_train = np.concatenate((X[0:u], X[v:num_samples]), axis=0)
        Y_test = Y[u:v]
        Y_train = np.concatenate((Y[0:u], Y[v:num_samples]), axis=0)
        u = u + length_test
        v = v + length_test
        accuracy, cm, classes = knn(X_train, Y_train, X_test, Y_test, k, metric)

        accuracies.append(accuracy)
        confusion_matrices.append(cm)

        print("=>Accuracy: ", accuracy)
        print("=>Confusion Matrix:\n "
              "{0}\n"
              "{1}\n".format(classes, cm))

    #print(accuracies)
    average_accuracy = sum(accuracies) / len(accuracies)
    sum_cm = 0
    for cm in confusion_matrices:
        sum_cm = sum_cm + cm
    average_cm = sum_cm/len(confusion_matrices)

    return average_accuracy, average_cm, classes


def main():
    parser = argparse.ArgumentParser(description = 'KNN Classifier')
    parser.add_argument('-k', help='Number of nearest neighbors', type=int, required=True)
    parser.add_argument('-m', help='Distance/Similarity Metric',type=str, required=True)
    args = parser.parse_args()
    k = args.k
    metric = args.m


    print("Loading Dataset...")
    X, Y = readData("iris.data")
    print("Completed!")
    accuracy, cm, classes = knnWithCrossValidation(X,Y, k=k, cv=5, metric=metric.lower())
    print("***KNN Classifier (with 5 fold cross validation)***")
    print("Parameters:"
          "\n\tNearest Neighbors (k): {0}"
          "\n\tDistance Metric : {1}".format(k, metric))
    print("=>Overall accuracy: ", accuracy)
    print("=>Overall Confusion Matrix:\n "
          "{0}\n"
          "{1}\n".format(classes, cm))
    print("Plotting Confusion Matrix...")
    plotConfusionMatrix(cm, classes,
                        normalize=True,
                        title="Confusion Matrix: KNN Classifier")

if __name__ == '__main__':
    main()
