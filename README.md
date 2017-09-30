# DeepMMD-GAN

Yet another approach for adversarial training.

- GANs are adversarial networks, no need to introduce those.
- MMD is maximum mean discrepancy, it is based on a very simple idea of how to detect difference between distributions.
  A decade ago MMD was very popular, notable property of MMD (compared to other tests) is that it works well when kernel is

MMD was already combined with GANs, see e.g.

- [Generative Moment Matching Networks](https://arxiv.org/abs/1502.02761)
- [Generative models and model criticism via optimized MMD](https://arxiv.org/pdf/1611.04488.pdf)


In these experiments something closer to GANs was considered, because "discriminator" tries to find
appropriate mapping to maximize MMD.


## Technical details

Using [pytorch](https://pytorch.org) for implementation, I am building upon DCGAN (official implementation from pytorch repo).

Discriminator also contains BatchNormalization as the last layer to ensure the scale of output.


## Results

- I have observed no divergences when experimenting
- at the same time quality of produced images isn't awesome
- results for different sizes of projections, this was removed from the notebooks.