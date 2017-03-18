import numpy as np
import util


def init_matplot_lib():
    global plt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.clf()


def plot_confusion_matrix(cm, classes):
    init_matplot_lib()
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
    init_matplot_lib()

    def conditional_plot(precision, recall, color):
        if len(precision) > 1:
            recall.append(0)
            precision.append(1)
            plt.plot(recall, precision, lw=2, c=color, label=class_name)
        else:
            plt.scatter(recall, precision, c=[color], label=class_name)

    from matplotlib.pyplot import cm
    colors = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    for class_name, color in zip(classes, colors):
        recall = map(lambda x: x[3], class_data[class_name])
        precision = map(lambda x: x[2], class_data[class_name])
        conditional_plot(precision, recall, color)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall for ' + metric + ' F1')
    plt.legend(loc="lower left")
    plt.show()
    plt.savefig(util.get_plot_path('precision-recall-' + metric))
