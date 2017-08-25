import numpy as np
import util
import sklearn.preprocessing


def init_matplot_lib():
    global plt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # plt.style.use('classic')
    # matplotlib.rcParams.update({'font.size': 17})
    plt.clf()


def plot_confusion_matrix(
        cm, classes, title="", normalize=True, cmap='Blues'):
    init_matplot_lib()

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
        util.get_plot_path('confusion-' + title),
        dpi=400,
        format='pdf',
        bbox_inches='tight')


def plot_roc(classes, class_data, metric):
    init_matplot_lib()

    def conditional_plot(fprs, tprs, color):
        if len(fprs) > 1:
            plt.plot(fprs, tprs, lw=2, c=color, label=class_name)
        else:
            plt.scatter(fprs, tprs, c=[color], label=class_name)

    from matplotlib.pyplot import cm
    colors = iter(cm.rainbow(np.linspace(0, 1, len(classes))))
    for class_name, color in zip(classes, colors):
        fprs = class_data[class_name][2]
        tprs = class_data[class_name][3]
        plt.plot(fprs, tprs, lw=2, c=color, label=class_name)
        #conditional_plot(fprs, tprs, color)

    plt.xlim(0, 0.5)
    plt.ylim(0.5, 1)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    plt.savefig(util.get_plot_path('roc-' + metric),
        dpi=400,
        format='pdf',
        bbox_inches='tight')


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
