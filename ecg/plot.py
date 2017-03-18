import numpy as np
import util


def plot_confusion_matrix(cm, classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(util.get_plot_path('confusion'))


def plot_precision_recall(classes, class_data, metric):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.clf()
    for class_name in classes:
        recall = map(lambda x: x[3], class_data[class_name])
        recall.append(0)
        precision = map(lambda x: x[2], class_data[class_name])
        precision.append(1)
        plt.plot(recall, precision, lw=2, label=class_name)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall for ' + metric + ' F1')
    plt.legend(loc="lower left")
    plt.show()
    plt.savefig(util.get_plot_path('precision-recall-' + metric))
