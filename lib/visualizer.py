""" This file contains Visualizer class based on Facebook's visdom.

Returns:
    Visualizer(): Visualizer class to display plots and images
"""

##
import os
import time
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

##
class Visualizer():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """
    # pylint: disable=too-many-instance-attributes
    # Reasonable.

    ##
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##
    def plot_current_errors(self, epoch, counter_ratio, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """

        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=4
        )

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None:
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=5
        )

    ##
    def print_current_errors(self, epoch, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
            batch_i (int): Current batch
            batch_n (int): Total Number of batches.
        """
        # message = '   [%d/%d] ' % (epoch, self.opt.niter)
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    ##
    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)
        message += 'max ' + self.opt.metric + ': %.3f' % best

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def display_current_images(self, reals, fakes, fixed):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        fixed = self.normalize(fixed.cpu().numpy())

        self.vis.images(reals, win=1, opts={'title': 'Reals'})
        self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        self.vis.images(fixed, win=3, opts={'title': 'Fixed'})

    def save_current_images(self, epoch, reals, fakes, fixed):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """

        vutils.save_image(reals, '%s/reals.png' % self.img_dir, normalize=True)
        vutils.save_image(fakes, '%s/fakes.png' % self.img_dir, normalize=True)
        vutils.save_image(fixed, '%s/fixed_fakes_%03d.png' %(self.img_dir, epoch+1), normalize=True)


        # import torch, cv2
        # residule = torch.abs(reals - fakes)
        # residule = residule.detach().cpu().numpy()
        # residule = (residule - np.min(residule)) / (np.max(residule) - np.min(residule)) * 255
        # residule = residule.astype("uint8")
        # residule = np.transpose(residule[0], [1, 2, 0])
        # if residule.shape[2] == 3:
        #     residule = cv2.cvtColor(residule, cv2.COLOR_BGR2GRAY)
        # elif residule.shape[2] == 1:
        #     residule = residule[:, :, 0]
        # residule = cv2.applyColorMap(residule, cv2.COLORMAP_OCEAN)
        # vutils.save_image(torch.cat((reals, fakes, residule), dim=0), '%s/residule.png' % self.img_dir, normalize=True)

    def visiualize_distribution(self, output_path, gt, scores_thres, scores, xlabel):

        overkill_n, underkill_n, tp_n, tn_n, t_n = 0, 0, 0, 0, 0
        # inference
        an_class = scores.copy()
        an_class[an_class < scores_thres] = 0
        an_class[an_class >= scores_thres] = 1  # anomaly


        # normal
        nor_index = np.argwhere(gt == 0).reshape(-1)
        for i in nor_index:
            if an_class[i] == 0:
                t_n += 1
                tp_n += 1
            else:
                overkill_n += 1

        # anomaly
        anom_index = np.argwhere(gt == 1).reshape(-1)
        for i in anom_index:
            if an_class[i] == 1:
                t_n += 1
                tn_n += 1
            else:
                underkill_n += 1
        n = an_class.shape[0]
        overkill_rate = 100*overkill_n/n
        underkill_rate = 100*underkill_n/n
        tp_rate = 100*tp_n/n
        tn_rate = 100*tn_n/n
        t_rate = 100*t_n/n
        title = 'Acc={:.1f} ; Overkill={:.1f} ; Underkill={:.1f} ; TP={:.1f} ; TN={:.1f}'.format(t_rate, overkill_rate, underkill_rate, tp_rate, tn_rate)
        print(title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.axvline(scores_thres, c='b')
        if len(nor_index) > 0:
            plt.scatter(scores[nor_index], nor_index, c='g')
        if len(anom_index) > 0:
            plt.scatter(scores[anom_index], anom_index, c='r')
        # plt.show()
        plt.savefig(output_path)
        plt.close()