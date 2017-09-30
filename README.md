# DeepMMD-GAN

Yet another approach for adversarial training.

- GANs are [adversarial networks](https://en.wikipedia.org/wiki/Generative_adversarial_network), no need to introduce those.
- MMD is maximum mean discrepancy (see e.g. [this presentation](http://alex.smola.org/teaching/iconip2006/iconip_3.pdf) by A. Smola),
  it is based on a very simple idea of how to detect difference between distributions.

  A decade ago MMD was very popular, notable property of MMD (compared to other tests) is that it works well when kernels.

MMD was already combined with GANs, see e.g.

- [Generative Moment Matching Networks](https://arxiv.org/abs/1502.02761)
- [Generative models and model criticism via optimized MMD](https://arxiv.org/pdf/1611.04488.pdf)


In my experiments something closer to traditional GANs was considered, because "discriminator" tries to find
appropriate mapping to maximize MMD.


## Technical details

Using [pytorch](https://pytorch.org) for implementation, I am building upon DCGAN (official implementation from pytorch repo).

Discriminator also contains BatchNormalization as the last layer to ensure the scale of output, that's quite critical.

Probably, it is among shortest implementations of GANs.


## Results and observations

I haven't done much tuning, but here what was found

- I have observed no divergences when experimenting, things are rather stable
  (at the same time at 64*64 definitely buggy generated pictures are appearing)
- at the same time quality of produced images isn't awesome
- results for different sizes of projections (2 ** 8 to 2 ** 15) aren't very different, while latter are quite slow
  (this was removed from the final version)
