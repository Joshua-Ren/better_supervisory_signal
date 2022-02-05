# Better Supervisory Signals by Observing Learning Paths
This repository is a simple implementation of [Better Supervisory Signals by Observing Learning Paths](https://openreview.net/forum?id=Iog0djAdbHj) (ICLR 2022).

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

### Experiments on toy Gaussian
We provide four co-lab notebooks in the *notebook* folder, you can copy it to co-lab or run on your local machine.
- **Part1_noisylabel**: some experiments verifying hypothesis 1
- **Part2_learningdynamics**: results in Fig.2 and 3, demonstrating why we should focus on hard samples
- **Part3_NTK**: support of proposition 1, more interesting facts of the gradients
- **CIFAR10_path_analysis**: analyzing the learning path on CIFAR10, verifying filter can help in both noisy and clean label case.
  - For this notebook, we need to prepare learning paths by running `cifar10h_gen_path.py` or `main_gen_path.py`
  - You can also [download](https://drive.google.com/file/d/17BKaG1er_b553Ik6_BUoipA6dlPf3sHq/view?usp=sharing) the results and put them under the correct path to run the experiments

### Experiments on Filter-KD
Actually, the Filter-KD algorithm is quite easy to understand and implement. Our motivation for this algorithm is to verify the analysis proposed in the paper. Hence this implementation might not be quite perfect and efficient. Here are some explanations of our implementation.
- **Dataloader**: in `utils.py`, we change the data loader to assign a constant index for all 50k training samples in CIFAR. We also add label noise in this loader. Hence the return of a loader becomes (x, y, ny, idx), where ny is the noisy label and idx is the assigned index. We use this index to update our p_smoothing table during training.
- **Tiny ImageNet**: after [downloading](http://cs231n.stanford.edu/tiny-imagenet-200.zip) this dataset, we need to change the image's path, `./data/refolder_tiny_imagenet.py` is for this
- **Generate teacher**: 

