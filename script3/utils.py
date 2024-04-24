import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
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


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Encoder_Decoder(nn.Module):
    def __init__(self, input_channels, generator_outputs_channels, ngf=4, skip=True):
        super(Encoder_Decoder, self).__init__()
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

        layers = [nn.Conv2d(input_channels, ngf,
                            kernel_size=4, stride=2, padding=1)]

        in_channels = ngf
        for out_channels in layer_specs:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(in_channels, out_channels,
                          kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

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

        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(
            in_channels, generator_outputs_channels, kernel_size=4, stride=2, padding='same'))
        layers.append(nn.Softmax(dim=1))

        self.decoder_layers = nn.ModuleList([
            *layers
        ])

    def forward(self, x):
        # Encoder
        layers = []
        for layer in self.encoder_layers:
            x = layer(x)
            layers.append(x)

        layers.append(x.view(-1, 1, 4, x.get_shape().as_list()[-1]))
        num_encoder_layers = len(layers)

        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            skip_layer = num_encoder_layers - i - 2
            if self.skip and i > 0:
                x = torch.cat([x, layers[skip_layer]], dim=3)
            x = layer(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, ndf, stride=2):
        super(Discriminator, self).__init__()
        self.n_layers = 3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                               stride=stride, padding=0, bias=True)  # 注意这里的padding设置为0
        self.convs = []
        in_channels = 0
        for i in range(self.n_layers):
            in_channels = out_channels
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == self.n_layers - 1 else 2
            self.convs.append([nn.Conv2d(in_channels, out_channels, kernel_size=4,
                              stride=stride, padding=0, bias=True), nn.BatchNorm2d(out_channels)])
        in_channels = out_channels
        self.last_conv = nn.Conv2d(
            in_channels, 1, kernel_size=4, stride=1, padding=0, bias=True)

    def forward(self, x):
        x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        x = self.conv1(x_padded)
        x = F.leaky_relu(x, 0.2)
        for i in range(self.n_layers):
            conv = self.convs[i][0]
            batchnorm = self.convs[i][1]
            x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
            x = conv(x_padded)
            x = batchnorm(x)
            x = F.leaky_relu(x, 0.2)
        x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        x = self.last_conv(x_padded)
        return torch.sigmoid(x)


class UNet(nn.Module):
    def __init__(self, input_channels, generator_outputs_channels, ngf=4, skip=True):
        super(UNet, self).__init__()
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
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0,))
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
            int(in_channels / 2), generator_outputs_channels, kernel_size=4, stride=2, padding=1, output_padding=0))
        layers.append(nn.Tanh())
        self.upsamle_layers = nn.ModuleList([
            *layers
        ])

    def forward(self, x):
        # Encoder
        layers = []
        x = x.permute(0, 3, 1, 2)
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


def unet(generator_inputs, generator_outputs_channels, ngf=4, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    in_channels = generator_inputs.shape[-1]
    net = UNet(input_channels=in_channels,
               generator_outputs_channels=generator_outputs_channels, ngf=ngf)
    return init_net(net, init_type, init_gain, gpu_ids)


def encoder_decoder(generator_inputs, generator_outputs_channels, ngf=4, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    in_channels = generator_inputs.shape[-1]
    net = Encoder_Decoder(in_channels, generator_outputs_channels, ngf)
    return init_net(net, init_type, init_gain, gpu_ids)


def discriminator(generator_inputs, generator_outputs_channels, ngf=4, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    in_channels = generator_inputs.shape[-1]
    net = Encoder_Decoder(in_channels, generator_outputs_channels, ngf)
    return init_net(net, init_type, init_gain, gpu_ids)
