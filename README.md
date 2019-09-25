##KNN and Naive Bayes Classifiers from scratch

This repository contains implementation of two popular classifiers in machine learning: KNN and Naive Bayes Classifier, 
without using any machine learning libraries or toolkits, while numpy and pandas are used for mathematical functions and 
data manipulation.
The evaluation is performed with 5 fold cross validation and each of the classifier outputs average accuracy and plots
the average confusion matrix.
### Instructions on how to run the program \
First, the following dependencies need to be installed:
```bash
$ sudo apt-get install python3-pip
$ pip3 install numpy
$ pip3 install pandas
$ pip3 install matplotlib
```

Then run the script
```bash
$ python3 knn_4808133.py -k 5 -m Euclidean
$ python3 knn_4808133.py -k 5 -m Cosine
$ python3 bayes_4808133.py
```

### Sample Ouput
\*\*\*KNN Classifier (with 5 fold cross validation)\*\*\*\
Parameters:
   * Nearest Neighbors (k): 5
   * Distance Metric : euclidean

![html dark](https://github.com/sdevkota007/KNN-And-NaiveBayes-Classifier/blob/master/screenshots/cm.png)
