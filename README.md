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

- Hard samples drift more in a long-enough train

<div align=center><img src="https://github.com/Joshua-Ren/better_supervisory_signal/blob/main/gifs/settings.gif" width="480"/><img src="https://github.com/Joshua-Ren/better_supervisory_signal/blob/main/gifs/hardsamples.gif" width="220"/></div>

- Hard samples converges slower in a **zig-zag** path

<div align=center>
<img src="https://github.com/Joshua-Ren/better_supervisory_signal/blob/main/gifs/zigzag_fast.gif" width="600" height="600" />
</div>

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
- **Tiny ImageNet**: after [downloading](http://cs231n.stanford.edu/tiny-imagenet-200.zip) (or `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`) this dataset, we need to change the image's path, `./data/refolder_tiny_imagenet.py` is for this
- **Generate teacher**: you can run `main_gen_teacher.py` to train a network under the supervision of one-hot labels. After training, the code will save a checkpoint and a \*.npy file. The checkpoint is the teacher for standard KD while the \*.npy file the the teacher for Filter-KD (i.e., q_smooth in our algorithm).
- **Distillation**: you can run `main_distill.py` to get results for standard KD or Filter-KD by specifying `--teach_type` (net is standard KD and table is Filter-KD).
- **Same initialization**: to make a fair comparison, you can run `old_cifar_noisy.py` in which the initialization of different methods would be exactly the same

# Reference
For technical details and full experimental results, please check [our paper](https://openreview.net/forum?id=Iog0djAdbHj).
```
@inproceedings{ren:zigzag,
    author = {Yi Ren and Shangmin Guo and Danica J. Sutherland},
    title = {Better Supervisory Signals by Observing Learning Paths},
    year = {2022},
    booktitle = {International Conference on Learning Representations (ICLR)},
    url = {https://openreview.net/forum?id=Iog0djAdbHj},
}
```

# Contact
Please contact renyi.joshua@gmail.com if you have any question on the codes.
