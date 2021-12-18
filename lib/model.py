"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import cv2


from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate
from torchsummary import summary


class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.compare_w = 12
        self.compare_h = 8
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

    ##
    def set_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        # self.fixed_input = self.fixed_input.unsqueeze(1)    # james
        # print(1, self.fixed_input.shape)
        # if self.fixed_input.dim() == 5:
        #     self.fixed_input = self.fixed_input[:, :, 0, :, :]
        # print(2, self.fixed_input.shape)
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """
        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'Weights_epoch_' + str(self.epoch+1))
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)
        torch.save({'epoch': epoch + 1, 'latent_thres': self.latent_thres, 'state_dict': self.netg.state_dict(), 'optimizer_state_dict': self.optimizer_g.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict(), 'optimizer_state_dict': self.optimizer_d.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """
        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize
            self.set_input(data)
            self.optimize_params()
            # if self.total_steps % self.opt.print_freq == 0:
            #     errors = self.get_errors()
            #     if self.opt.display:
            #         counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
            #         self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)
            #
            # if self.total_steps % self.opt.save_image_freq == 0:
            #     reals, fakes, fixed = self.get_current_images()
            #     self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
            #     if self.opt.display:
            #         self.visualizer.display_current_images(reals, fakes, fixed)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        print("Pass rate: %d", self.pass_rate)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            self.find_threshold()
            # self.val_NG_distance()
            self.model_scores()
            if self.epoch % self.opt.save_model_freq == 0:
                self.save_weights(self.epoch)
        print(">> Training model %s.[Done]" % self.name)

    ##

    def model_scores(self):
        PASS_label, PASS_scores = self.test(testing_type="PASS")
        NG_label, NG_scores = self.test(testing_type="NG")
        # NG_label, NG_scores = self.test(testing_type="NG")
        total_label = torch.cat((PASS_label, NG_label)).detach().cpu().numpy()
        total_scores = torch.cat((PASS_scores, NG_scores)).detach().cpu().numpy()
        dst = os.path.join(self.opt.outf, self.opt.name, "test", 'distribution')
        if not os.path.isdir(dst):
            os.makedirs(dst)
        xlabel = 'Train Pass={:.1f} ; Val Pass={:.1f} ; Testing Pass={:.1f} ; Testing NG={:.1f}'.format(len(self.dataloader['train'].dataset), len(self.dataloader['val'].dataset), len(self.dataloader['test_PASS'].dataset), len(self.dataloader['test_NG'].dataset))
        self.visualizer.visiualize_distribution(output_path=dst + "/" + str(self.epoch+1) + ".png", gt=total_label, scores_thres=self.latent_thres.detach().cpu().numpy(), scores=total_scores, xlabel=xlabel)

    def find_threshold(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """

        with torch.no_grad():
            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['val'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['val'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['val'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['val'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            if self.an_scores.shape[0] < self.quantity_val:
                print("Val data must be greater than ", self.quantity_val)
                exit()

            self.times = []
            self.total_steps = 0
            epoch_iter = 0

            for i, data in enumerate(self.dataloader['val'], 0):

                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                distance = torch.pow((self.input-self.fake), 2)
                distance = distance.reshape(distance.shape[0], -1)
                error = torch.mean(distance, dim=1)
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)
                time_o = time.time()
                self.times.append(time_o - time_i)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)
            self.latent_thres = self.an_scores.sort()[0][round(self.pass_rate * self.an_scores.shape[0])]

    # def val_NG_distance(self):
    #
    #     with torch.no_grad():
    #         # Create big error tensor for the test set.
    #         self.an_scores = torch.zeros(size=(len(self.dataloader['val_NG'].dataset),), dtype=torch.float32, device=self.device)
    #
    #         if self.an_scores.shape[0] < self.quantity_val:
    #             print("Val data must be greater than ", self.quantity_val)
    #             exit()
    #
    #         for i, data in enumerate(self.dataloader['val_NG'], 0):
    #
    #             self.set_input(data)
    #             self.fake, latent_i, latent_o = self.netg(self.input)
    #             error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
    #             self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
    #
    #         self.max_val_NG_distance = torch.mean(self.an_scores)

    def test(self, testing_type='PASS'):
        with torch.no_grad():
            if testing_type == 'PASS':
                loader_type = 'test_PASS'
                true_label = 0
                mismatch_folder = 'Overkill'
            elif testing_type == 'NG':
                loader_type = 'test_NG'
                true_label = 1
                mismatch_folder = 'Underkill'


            # Create big error tensor for the test set.
            image_set = torch.zeros(size=(3, len(self.dataloader[loader_type].dataset), self.opt.nc, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
            self.an_scores = torch.zeros(size=(len(self.dataloader[loader_type].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader[loader_type].dataset),), dtype=torch.float32, device=self.device) + true_label
            self.latent_i = torch.zeros(size=(len(self.dataloader[loader_type].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader[loader_type].dataset), self.opt.nz), dtype=torch.float32, device=self.device)


            self.times = []
            self.total_steps = 0
            epoch_iter = 0

            for i, data in enumerate(self.dataloader[loader_type], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                # ti = time.time()
                self.fake, latent_i, latent_o = self.netg(self.input)
                # tf = time.time()
                # print("Testing time = ", tf - ti)

                distance = torch.pow((self.input - self.fake), 2)
                distance = distance.reshape(distance.shape[0], -1)
                error = torch.mean(distance, dim=1)
                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)
                time_o = time.time()
                self.times.append(time_o - time_i)
                real, fake, _ = self.get_current_images()
                image_set[0, i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = real.clone()
                image_set[1, i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = fake.clone()
                image_set[2, i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = torch.abs(real - fake).clone()
            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)
            an_class = self.an_scores.clone()
            an_class[an_class < self.latent_thres] = 0
            an_class[an_class >= self.latent_thres] = 1  # anomaly

            if not self.opt.isTrain:
                # Save test result.
                dst = os.path.join(self.opt.outf, self.opt.name, "test", 'images')
                if not os.path.isdir(dst):
                    os.makedirs(dst)
                mismatch_path = os.path.join(self.opt.outf, self.opt.name, "test", 'epoch_'+str(self.epoch+1) + "_" + mismatch_folder)
                if not os.path.isdir(mismatch_path):
                    os.makedirs(mismatch_path)
                n = 0
                while n <= image_set.shape[1]:
                    # Detail
                    compare = torch.cat((image_set[0, n, 0:1], image_set[1, n, 0:1], image_set[2, n, 0:1]), dim=0)
                    if self.gt_labels[n] != an_class[n]:
                        vutils.save_image(compare.unsqueeze(1), '%s/Scores_%03f_Index_%04d.png' % (mismatch_path, self.an_scores[n], n), normalize=True, nrow=3)
                    for j in range(n+1, n + int(round(self.compare_w*self.compare_h)/3)):
                        if j < image_set.shape[1]:
                            mini_compare = torch.cat((image_set[0, j, 0:1], image_set[1, j, 0:1], image_set[2, j, 0:1]))  # .unsqueeze(1)
                            if self.gt_labels[j] != an_class[j]:
                                vutils.save_image(mini_compare.unsqueeze(1), '%s/Scores_%03f_Index_%04d.png' % (mismatch_path, self.an_scores[j], j), normalize=True, nrow=3)
                            compare = torch.cat((compare, mini_compare), dim=0)
                    compare = compare.unsqueeze(1)
                    vutils.save_image(compare, '%s/Compare_%03d_%s_epoch%d_%d.png' % (dst, i + 1, testing_type, self.epoch + 1, n), normalize=True, nrow=self.compare_w)
                    n += int(round(self.compare_w*self.compare_h)/3)
        return self.gt_labels.clone(), self.an_scores.clone()

##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)
        # -- Misc attributes
        self.epoch = 0
        self.quantity_val = 10
        self.pass_rate = self.opt.pass_rate
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        print(summary(self.netg, (opt.nc, opt.isize, opt.isize)))
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.load_weights:
            print("\nLoading pre-trained networks.")
            pathG = "./output/{}/{}/train/Weights_epoch_{}/netG.pth".format(self.name.lower(), self.opt.dataset, self.opt.load_weights)
            pathD = "./output/{}/{}/train/Weights_epoch_{}/netD.pth".format(self.name.lower(), self.opt.dataset, self.opt.load_weights)
            print("pathG = ", pathG)
            self.netg.load_state_dict(torch.load(pathG)['state_dict'])
            self.optimizer_g.load_state_dict(torch.load(pathG)['optimizer_state_dict'])
            print("\tDone.\n")
            print("pathD = ", pathD)
            self.netd.load_state_dict(torch.load(pathD)['state_dict'])
            self.optimizer_d.load_state_dict(torch.load(pathD)['optimizer_state_dict'])
            print("\tDone.\n")
            self.opt.iter = torch.load(pathG)['epoch']
            self.epoch = torch.load(pathG)['epoch']-1
            self.latent_thres = torch.load(pathG)['latent_thres']
        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()  # reconstruction loss
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        # self.input = self.input[:, 0, :, :]
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()


    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()
