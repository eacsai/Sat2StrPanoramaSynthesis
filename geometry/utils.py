import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
import functools


to_pil = transforms.ToPILImage()

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('ConvTranspose2d') != -1 and m.bias is not None:
            init.normal_(m.weight.data, 0.0, 0)
            init.constant_(m.bias.data, 0.0)
            m.bias.data[-1] = 1.0
            
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # if len(gpu_ids) > 0:
    #     assert (torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    net.to('cuda')
    print('initialize network %s' % net.__class__.__name__)
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Encoder_Decoder(nn.Module):
    def __init__(self, input_channels, generator_outputs_channels = 64, ngf=4, skip=True):
        super(Encoder_Decoder, self).__init__()
        self.skip = skip
        self.generator_outputs_channels = generator_outputs_channels
        # Encoder layers
        # Using list comprehension to create subsequent encoder layers
        layer_specs = [
            # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
            ngf * 2,
            # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
            ngf * 4,
            # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
            ngf * 8,
            # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
            ngf * 8,
            # encoder_6: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            ngf * 8,
            # encoder_7: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            ngf * 8,
            # ngf * 8, # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]

        layers = [nn.ModuleList([nn.Conv2d(input_channels, ngf,
                                           kernel_size=4, stride=2, padding=1)])]

        in_channels = ngf
        for out_channels in layer_specs:
            encode_layers = []
            encode_layers.append(nn.LeakyReLU(0.2, inplace=True))
            encode_layers.append(nn.Conv2d(in_channels, out_channels,
                                           kernel_size=4, stride=2, padding=1))
            encode_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            layers.append(nn.ModuleList([
                *encode_layers
            ]))

        self.encoder_layers = nn.ModuleList([
            *layers
        ])

        # Decoder layers
        layers = []
        layer_specs = [
            # (ngf * 8, 0.5),   # decoder_8: [batch, 1, 4, ngf * 8] => [batch, 2, 8, ngf * 8 * 2]
            # decoder_7: [batch, 2, 8, ngf * 8 * 2] => [batch, 4, 16, ngf * 8 * 2]
            (ngf * 8, 0.0),
            # decoder_6: [batch, 4, 16, ngf * 8 * 2] => [batch, 8, 32, ngf * 8 * 2]
            (ngf * 8, 0.0),
            # decoder_5: [batch, 8, 32, ngf * 8 * 2] => [batch, 16, 64, ngf * 8 * 2]
            (ngf * 8, 0.0),
            # decoder_4: [batch, 16, 64, ngf * 8 * 2] => [batch, 32, 128, ngf * 4 * 2]
            (ngf * 4, 0.0),
            # decoder_3: [batch, 32, 128, ngf * 4 * 2] => [batch, 64, 256, ngf * 2 * 2]
            (ngf * 2, 0.0),
            # decoder_2: [batch, 64, 256, ngf * 2 * 2] => [batch, 128, 512, ngf * 2 * 2]
            (ngf, 0.0),
        ]

        for index, (out_channels, dropout) in enumerate(layer_specs):
            decoder_layers = []
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            layers.append(nn.ModuleList([
                *decoder_layers
            ]))
            in_channels = out_channels * 2

        self.decoder_layers = nn.ModuleList([
            *layers
        ])

        # self.relu = nn.ReLU(inplace=True)
        # self.conv_transpose = nn.ConvTranspose2d(
        #     int(in_channels / 2), generator_outputs_channels, kernel_size=4, stride=2, padding=1, output_padding=0)
        # self.softmax = nn.Softmax(dim=1)
        
        # # 初始化权重为0
        # nn.init.constant_(self.conv_transpose.weight.data, 0)
        # init.constant_(self.conv_transpose.bias.data, 0.0)
        # self.conv_transpose.bias.data[-1] = 1.0
        
        layers = []
        
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(
            int(in_channels / 2), generator_outputs_channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True))
        layers.append(nn.Softmax(dim=1))
        self.upsample_layers = nn.ModuleList([
            *layers
        ])

    def forward(self, x):
        # Encoder
        layers = []
        for encoder_layers in self.encoder_layers:
            for layer in encoder_layers:
                x = layer(x)
            layers.append(x)

        layers.append(x.view(-1, x.shape[1], 1, 4))
        num_encoder_layers = len(layers)

        # Decoder
        for i, decoder_layers in enumerate(self.decoder_layers):
            skip_layer = num_encoder_layers - i - 2
            if self.skip and i > 0:
                x = torch.cat([x, layers[skip_layer]], dim=1)
            for decode in decoder_layers:
                x = decode(x)
            layers.append(x)
        
        # unsample
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        return x


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Generator(nn.Module):
    def __init__(self, input_channels, generator_outputs_channels, ngf=4, skip=True):
        super(Generator, self).__init__()
        self.skip = skip
        # Encoder layers
        # Using list comprehension to create subsequent encoder layers
        layer_specs = [
            # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
            ngf * 2,
            # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
            ngf * 4,
            # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
            ngf * 8,
            # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
            ngf * 8,
            # encoder_6: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            ngf * 8,
            # encoder_7: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            ngf * 8,
            # ngf * 8, # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]

        layers = [nn.ModuleList([nn.Conv2d(input_channels, ngf,
                                           kernel_size=4, stride=2, padding=1)])]

        in_channels = ngf
        for out_channels in layer_specs:
            encode_layers = []
            encode_layers.append(nn.LeakyReLU(0.2, inplace=True))
            encode_layers.append(nn.Conv2d(in_channels, out_channels,
                                           kernel_size=4, stride=2, padding=1))
            encode_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            layers.append(nn.ModuleList([
                *encode_layers
            ]))

        self.encoder_layers = nn.ModuleList([
            *layers
        ])

        # Decoder layers
        layers = []
        layer_specs = [
            # (ngf * 8, 0.5),   # decoder_8: [batch, 1, 4, ngf * 8] => [batch, 2, 8, ngf * 8 * 2]
            # decoder_7: [batch, 2, 8, ngf * 8 * 2] => [batch, 4, 16, ngf * 8 * 2]
            (ngf * 8, 0.5),
            # decoder_6: [batch, 4, 16, ngf * 8 * 2] => [batch, 8, 32, ngf * 8 * 2]
            (ngf * 8, 0.5),
            # decoder_5: [batch, 8, 32, ngf * 8 * 2] => [batch, 16, 64, ngf * 8 * 2]
            (ngf * 8, 0.5),
            # decoder_4: [batch, 16, 64, ngf * 8 * 2] => [batch, 32, 128, ngf * 4 * 2]
            (ngf * 4, 0.0),
            # decoder_3: [batch, 32, 128, ngf * 4 * 2] => [batch, 64, 256, ngf * 2 * 2]
            (ngf * 2, 0.0),
            # decoder_2: [batch, 64, 256, ngf * 2 * 2] => [batch, 128, 512, ngf * 2 * 2]
            (ngf, 0.0),
        ]

        for index, (out_channels, dropout) in enumerate(layer_specs):
            decoder_layers = []
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            layers.append(nn.ModuleList([
                *decoder_layers
            ]))
            in_channels = out_channels * 2

        self.decoder_layers = nn.ModuleList([
            *layers
        ])

        layers = []
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(
            int(in_channels / 2), generator_outputs_channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
        layers.append(nn.Tanh())
        self.upsamle_layers = nn.ModuleList([
            *layers
        ])

    def forward(self, x):
        # Encoder
        layers = []
        
        for encoder_layers in self.encoder_layers:
            for layer in encoder_layers:
                x = layer(x)
            layers.append(x)

        layers.append(x.view(-1, x.shape[1], 1, 4))
        num_encoder_layers = len(layers)

        # Decoder
        for i, decoder_layers in enumerate(self.decoder_layers):
            skip_layer = num_encoder_layers - i - 2
            if self.skip and i > 0:
                x = torch.cat([x, layers[skip_layer]], dim=1)
            for decode in decoder_layers:
                x = decode(x)

            layers.append(x)

        # unsample
        for i, upsample in enumerate(self.upsamle_layers):
            x = upsample(x)
        return x

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

def generator(in_channels, generator_outputs_channels, ngf=4, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    net = Generator(input_channels=in_channels,
               generator_outputs_channels=generator_outputs_channels, ngf=ngf).to('cuda')
    return init_net(net, init_type, init_gain, gpu_ids)

def encoder_decoder(in_channels, generator_outputs_channels, ngf=4, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    net = Encoder_Decoder(in_channels, generator_outputs_channels, ngf).to('cuda')
    return init_net(net, init_type, init_gain, gpu_ids)

def discriminator(in_channels, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    net = Discriminator(in_channels).to('cuda')
    return init_net(net, init_type, init_gain, gpu_ids)
