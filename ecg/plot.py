import numpy as np
import util
import sklearn.preprocessing


def init_matplot_lib():
    global plt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('classic')
    plt.clf()


def plot_confusion_matrix(
        cm, classes, title="", normalize=True, cmap='Blues'):
    init_matplot_lib()

    classes = [c if c != 'SUDDEN_BRADY' else 'CHB' for c in classes]

    cm = sklearn.preprocessing.normalize(cm, norm='l1', axis=1, copy=True)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.clim(0, 1)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(
        util.get_plot_path('confusion-' + title), dpi=400, format='pdf',
        bbox_inches='tight')


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


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(
            pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(
        AUC, title, xlabel, ylabel, xticklabels, yticklabels,
        figure_width=40, figure_height=20, correct_orientation=True,
        cmap='Blues'):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(
        AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        # ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(
        classification_report, title='Classification report', cmap='Blues'):
    '''
    Plot scikit-learn classification report.
    Extension based on http://stackoverflow.com/a/31689645/395857
    '''

    import evaluate
    init_matplot_lib()
    classes, plotMat, support, class_names = \
        evaluate.parse_classification_report(classification_report)
    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = True
    heatmap(
        np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels,
        figure_width, figure_height,
        correct_orientation=correct_orientation, cmap=cmap)
    plt.savefig(util.get_plot_path(title), dpi=200, bbox_inches='tight')
    plt.close()
