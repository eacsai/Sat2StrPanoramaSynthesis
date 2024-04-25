import argparse
import os
import torch
import time

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # parser.add_argument("--input_dir", help="path to folder containing images", default='facades/train')
        parser.add_argument("--dataset", help="dataset", default='CVACT')
        parser.add_argument("--mode", choices=["train", "test", "export"], default="train")
        parser.add_argument("--output_dir", help="where to put output files", default='pix2pix_perceploss')
        parser.add_argument("--seed", type=int)
        parser.add_argument("--checkpoint", help="directory with checkpoint to resume training from or use for testing")

        # parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
        parser.add_argument("--start_epochs", type=int, default=0, help="number of training epochs")
        parser.add_argument("--max_epochs", type=int, default=35, help="number of training epochs")
        parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
        parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
        parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
        parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

        parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
        parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
        parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
        parser.add_argument("--batch_size", type=int, default=4, help="number of images in batch")
        parser.add_argument("--which_direction", type=str, default="AtoG", choices=["AtoG", "GtoA"])
        parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
        parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
        parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
        parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
        parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
        parser.set_defaults(flip=True)
        parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
        parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")

        parser.add_argument("--inputs_type", choices=["original", "geometry", "polar", "tanpolar"], default="geometry")

        parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
        parser.add_argument("--l1_weight_grd", type=float, default=100.0, help="weight on GAN term for generator gradient")
        parser.add_argument("--l1_weight_aer", type=float, default=0.0, help="weight on L1 term for generator gradient")
        parser.add_argument("--perceptual_weight_grd", type=float, default=0.0, help="weight on GAN term for generator gradient")
        parser.add_argument("--perceptual_weight_aer", type=float, default=0.0, help="weight on GAN term for generator gradient")

        parser.add_argument("--heightPlaneNum", type=int, default=1, help="weight on GAN term for generator gradient")
        parser.add_argument("--radiusPlaneNum", type=int, default=32, help="weight on GAN term for generator gradient")
        parser.add_argument("--height_mode", choices=['radiusPlaneMethod', 'heightPlaneMethod'], default='radiusPlaneMethod')
        # Only when 'height_mode' is 'radiusPlaneMethod', the following two parameters are required. Otherwise not.
        parser.add_argument("--method", choices=['column', 'point'], default='column')
        parser.add_argument("--geoout_type", choices=['volume', 'image'], default='image')

        parser.add_argument("--finalout_type", choices=['image', 'rgba', 'fgbg'], default='image')

        parser.add_argument("--skip", type=int, default=0, help="use skip connection or not")

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # export options
        parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
        
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')


        # model parameters
        parser.add_argument('--model', type=str, default='pix2pix', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='unet_128', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--input_type', type=str, default='estimated_height', help='input image type for the generator')
        parser.add_argument('--model_type', type=str, default='primary', help='type for the generator')
        parser.add_argument('--results_dir', type=str, default='./results', help='save results')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        txt_name = time.time()
        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, isTrain=True):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = isTrain   # train or test

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
