# Better Supervisory Signals by Observing Learning Paths
This repository is the official implementation of [Better Supervisory Signals by Observing Learning Paths](https://openreview.net/forum?id=Iog0djAdbHj) (ICLR 2022).

### Abstract
Better supervision might lead to better performance. 
In this paper, we first clarify that good supervisory signal should be close to the ground truth p(y|x).
The success of label smoothing and self-distillation might partly owe to this.
To further answer why and how better supervision emerges, 
we observe the learning path, i.e., the trajectory of the network's predicted distribution, for each training sample during training.
We surprisingly find that the model can spontaneously refine *bad* labels with through a **zig-zag** learning path,
which occurs on both toy and real datasets.
Observing the learning path not only provides a new perspective for understanding knowledge distillation, overfitting, and learning dynamics, 
but also reveals that the supervisory signal of a teacher network can be very unstable near the best points in training on real tasks.
Inspired by this, we propose a new algorithm, Filter-KD, which further enhances classification performance, 
as verified by experiments in various settings.

### Interesting findings


# About this repo
We would first say sorry to one of the reviewers of this paper because we were eventually not able to accomplish the ImageNet experiments in this version, as GPU is really expensive T_T. We hope someone who is interested in this work could verify our results on a large dataset.

### Requirements
- This codebase is written for `python3`.
- `Pytorch` and `torchvision` are necessary, which version seems not a big deal.
- We use `wandb` to track the results.


