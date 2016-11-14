import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider

from db_constants import rhythm_code2name, rhythm_name2code


# Color codes (same as those used in the qa-tool)
not_specified = 'DarkCyan'
bg_color = 'LightSteelBlue'

rhythm_name2color = {
    'NSR': 'Blue', 'SVT': 'White', 'SUDDEN_BRADY': not_specified,
    'AVB_TYPE2': 'Chocolate', 'PAUSE': 'Magenta', 'AFIB': 'LimeGreen',
    'VT': 'OrangeRed', 'BIGEMINY': 'Purple', 'TRIGEMINY': 'LightPink',
    'VF': 'Gold', 'PACING': not_specified, 'WENCKEBACH': 'MediumPurple',
    'NOISE': 'LightGray'}


def pltWithButton(sig, fs=-1, start=0, window=-1, annotation_dict={}, rhy_dict={}):
    """
    Plot ECG signal, color-coded based on different rhythm types
    :param sig: raw signal
    :param fs: sampling frequency of sig
    :param start: starting time
    :param window: frame window length
    :param annotation_dict: a dictionary of PQRST peak locations
    :param rhy_dict: dictionary of rhythm types, with rhythm codes as keys
                     and list of (onset, offset) tuples as values for each key
    :return: Figure handles for interactive plotting
    """

    # set the font
    font = {'family': 'normal', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)

    # plotting constants:
    main_plt_bottom = 0.4
    # --> slider position: [left, bottom, width, height]
    slider_position = [0.125, main_plt_bottom*3./4, 0.7, main_plt_bottom/8.]

    # --> forward/backward and delineation button position:
    # [left, bottom, width, height]
    bbutton_position = [0.7, 0.03, 0.1, main_plt_bottom/8.]
    fbutton_position = [0.81, 0.03, 0.1, main_plt_bottom/8.]
    delin_button_position = [0.01, main_plt_bottom, 0.05, 0.05]

    fig, ax = plt.subplots()
    ax.set_axis_bgcolor(bg_color)

    # to turn off the rolling format for xaxis numbers
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.subplots_adjust(bottom=0.4)
    sig_length = len(sig)
    x = range(sig_length)

    # plot the signal
    if len(rhy_dict) > 0:
        for rhy_code in rhy_dict:
            for onset, offset in rhy_dict[rhy_code]:
                l, = ax.plot(x[onset:offset], sig[onset:offset],
                             color=rhythm_name2color[rhythm_code2name[rhy_code]],
                             lw=2)
    else:
        l, = ax.plot(x, sig)

    # Add slider
    # Create an axis object to hold the slider
    slider_ax = plt.axes(slider_position)
    slider = Slider(slider_ax, 'x-limits', valmin=0.0,
                    valmax=sig_length, valinit=0)

    annotation_list = annotation_dict.keys()
    plot_handles = dict()
    annotation_marker = {'R': 'r^', 'Q': 'g^', 'S': 'm^', 'P': 'b^'}

    for item in annotation_list:
        annotation = annotation_dict[item]
        h, = ax.plot(annotation, sig[annotation], annotation_marker[item])
        plot_handles[item] = h

    if fs > 0:
        x1 = np.int(start*fs)
        x2 = np.int(np.min([(start+window)*fs, sig_length]))
        y1 = np.min(sig[x1:x2])
        y2 = np.max(sig[x1:x2])
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)

        class ButtonProcessor():

            def __init__(self, axes, label):
                if label == 'Forward':
                    self.button = Button(axes, label)
                    self.button.on_clicked(self.forward)
                elif label == 'Backward':
                    self.button = Button(axes, label)
                    self.button.on_clicked(self.backward)
                elif '_N' in label:
                    if len(label) > 7:
                        self.button = Button(axes, label[:4]+label[-2:])
                    else:
                        self.button = Button(axes, label)

                    self.rhy_code = rhythm_name2code[label.split('_N')[0]]
                    self.rhy_int = rhy_dict[self.rhy_code]
                    self.button.on_clicked(self.nextInt)

                elif '_P' in label:
                    if len(label) > 7:
                        self.button = Button(axes, label[:4]+label[-2:])
                    else:
                        self.button = Button(axes, label)

                    self.rhy_code = rhythm_name2code[label.split('_P')[0]]
                    self.rhy_int = rhy_dict[self.rhy_code]
                    self.button.on_clicked(self.prevInt)

            def forward(self, event):
                x1, x2 = ax.get_xlim()
                x1 = np.min([x1+fs,sig_length-window*fs])
                x2 = np.min([x1+window*fs, sig_length])
                y1 = np.min(sig[x1:x2])
                y2 = np.max(sig[x1:x2])
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
                plt.draw()

            def backward(self, event):
                x1, x2 = ax.get_xlim()
                x1 = np.max([x1-fs,0])
                x2 = np.min([x1+window*fs, sig_length])
                y1 = np.min(sig[x1:x2])
                y2 = np.max(sig[x1:x2])
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
                plt.draw()

            def nextInt(self, event):
                x1, x2 = ax.get_xlim()
                for interval in self.rhy_int:
                    if interval[0] >= x2:
                        x1 = interval[0]
                        break

                x2 = np.min([x1+window*fs, sig_length])
                y1 = np.min(sig[x1:x2])
                y2 = np.max(sig[x1:x2])
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
                plt.draw()

            def prevInt(self, event):
                x1, x2 = ax.get_xlim()
                for interval in self.rhy_int[::-1]:
                    if interval[1] <= x1:
                        x1 = interval[0]
                        break

                x2 = np.min([x1+window*fs, sig_length])
                y1 = np.min(sig[x1:x2])
                y2 = np.max(sig[x1:x2])
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
                plt.draw()

        # setting the forward/backward button locations
        axbackward = plt.axes(bbutton_position)
        axforward = plt.axes(fbutton_position)

        # creating the forward/backward buttons
        bforward = ButtonProcessor(axforward, 'Forward')
        bbackward = ButtonProcessor(axbackward, 'Backward')

        # setting the location for delineation buttons
        rax_delin = plt.axes(delin_button_position)

        # creating the delineation buttons on the right side bar
        check = CheckButtons(rax_delin, tuple(annotation_list),
                             tuple([True for item in annotation_list]))

        # defining the actions to take place when changing delineation
        # button settings
        def func(label):
            label_handle = [item[1] for item in plot_handles.items() if label == item[0]]
            label_handle[0].set_visible(not label_handle[0].get_visible())
            plt.draw()

        # Define a function to run whenever the slider changes its value
        def update(val):
            x1 = val
            x2 = np.min([x1+window*fs, sig_length])
            y1 = np.min(sig[x1:x2])
            y2 = np.max(sig[x1:x2])
            ax.set_xlim(x1, x2)
            ax.set_ylim(y1, y2)
            plt.draw()

        # calling the function when delineation buttons are hit
        check.on_clicked(func)

        # Register the function update to run when the slider changes value
        slider.on_changed(update)

        # creating next/previous buttons for the non-NSR rhythms in the signal
        rhy_next_handles = {}
        rhy_prev_handles = {}
        for rhy_ind, rhy_key in enumerate(rhy_dict):
            if rhythm_code2name[rhy_key] != 'NSR':
                axprev = plt.axes([0.1+(rhy_ind%3)*0.2, 0.2-0.1*(rhy_ind//3),
                                   0.09, main_plt_bottom/8.])
                axnext = plt.axes([0.2+(rhy_ind%3)*0.2, 0.2-0.1*(rhy_ind//3),
                                   0.09, main_plt_bottom/8.])
                rhy_prev_handles[rhy_key] = \
                    ButtonProcessor(axprev, rhythm_code2name[rhy_key]+'_P')
                rhy_next_handles[rhy_key] = \
                    ButtonProcessor(axnext, rhythm_code2name[rhy_key]+'_N')

    plt.show()

    return fig, ax, bforward, bbackward, rhy_prev_handles, \
           rhy_next_handles, check, slider
