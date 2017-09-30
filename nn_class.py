import numpy
import torch
import torch.nn as nn


class GeneratorNet(nn.Module):
    def __init__(self, input_nfilters, generator_nfilters, image_size=64, n_colors=3):
        super(GeneratorNet, self).__init__()

        init_mult = image_size // 8
        layers = [nn.ConvTranspose2d(input_nfilters, generator_nfilters * init_mult, 4, 1, 0, bias=False),
                  nn.BatchNorm2d(generator_nfilters * init_mult),
                  nn.LeakyReLU(0.2, inplace=True)]

        for layer_i in range(int(numpy.log2(image_size // 8))):
            layers.append(nn.ConvTranspose2d(generator_nfilters * init_mult,
                                             generator_nfilters * init_mult // 2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(generator_nfilters * init_mult // 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            init_mult = init_mult // 2

        layers.append(nn.ConvTranspose2d(generator_nfilters * init_mult, n_colors, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class DiscriminatorNet(nn.Module):
    def __init__(self, discriminator_nfilters, image_size=64, n_colors=3):
        super(DiscriminatorNet, self).__init__()

        # input is (nc) x image_size x image_size
        init_mult = 1
        layer_mult = 2
        layers = [
            nn.Conv2d(n_colors, discriminator_nfilters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for layer_i in range(int(numpy.log2(image_size // 8))):
            layers.append(
                nn.Conv2d(discriminator_nfilters * init_mult, discriminator_nfilters * init_mult * layer_mult, 4, 2, 1,
                          bias=False))
            layers.append(nn.BatchNorm2d(discriminator_nfilters * init_mult * layer_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            init_mult *= layer_mult

        layers.append(
            nn.Conv2d(discriminator_nfilters * init_mult, discriminator_nfilters * init_mult * layer_mult, 4, 2,
                      padding=0, bias=False))
        layers.append(nn.BatchNorm2d(discriminator_nfilters * init_mult * layer_mult, affine=False))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        output = self.main(x)
        return output.view(output.size(0), -1)
