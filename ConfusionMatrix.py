from __future__ import division
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("Could Not import matplotlib")

def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title="Confusion Matrix",
                        cmap=None,
                        plot = True):
    '''
    plots the given confusion matrix
    :param cm: confusion matrix
    :param classes: list of unique classes
    :param normalize: True if elements in cm are of float type
    :param title: Title of the plot
    :param cmap:
    :param plot: True if a plot is to be generated
    :return:
    '''

    try:
        if cmap is None:
            cmap = plt.cm.Blues
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        if plot:
            plt.show()

    except Exception as e:
        print("Could not generate graphical plot, continuing anyway!", e)
        return None,classes, cm

    return ax,classes, cm

def confusionMatrix(actual, predicted):
    '''
    calculates the confusion matrix from true labels and predicted labels
    :param actual: true labels
    :param predicted: true labels
    :return: confusion matrix
    '''
    actual = list(actual)
    predicted = list(predicted)
    assert len(actual)==len(predicted)
    classes = np.unique(actual).tolist()
    num_classes = len(classes)
    cm = np.zeros((num_classes,num_classes), dtype=int)
    for i in range(len(predicted)):
        # actual label along the rows and predicted label along the column
        x = classes.index(actual[i])
        y = classes.index(predicted[i])
        cm[x][y] +=1
    return cm, classes



if __name__ == '__main__':
    # P = ['a', 'b', 'c', 'a', 'b']
    # A = ['a', 'b', 'b', 'c', 'a']
    A = ['Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-virginica', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
    P = ['Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa']

    cm, classes = confusionMatrix(A, P)
    plotConfusionMatrix(cm.astype(float), classes, normalize=True)
