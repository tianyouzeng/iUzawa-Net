# The iUzawa-Net for Nonsmooth Optimal Control of Linear PDEs

This repository contains the source code for the paper "**_Learning to Control: The iUzawa-Net for Nonsmooth Optimal Control of Linear PDEs_**" by Yongcun Song, Xiaoming Yuan, Hangrui Yue, and Tianyou Zeng.
The paper can be found at [arxiv:2602.12273](https://arxiv.org/abs/2602.12273).

## Requirements

To run the code in this repository, you will need following software and packages:

- [Python](https://www.python.org/) (==3.13.3)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyTorch](https://pytorch.org/) (==2.7.0, CUDA==12.8)

For some of the data generation scripts, the following library is required:

- [PyPardiso](https://github.com/haasad/PyPardiso)

An example conda environment is provided in the [`env`](https://github.com/tianyouzeng/iUzawa-Net/tree/main/env) folder.

## Files

The name of the files suggests their functionality. For example:

- [`elliptic_train.py`](https://github.com/tianyouzeng/iUzawa-Net/blob/main/elliptic_train.py) is the source code for training the iUzawa-Net for solving the elliptic optimal control problems in Section 6.1 of the [paper](https://arxiv.org/abs/2602.12273).
- [`elliptic_ic_train_shared.py`](https://github.com/tianyouzeng/iUzawa-Net/blob/main/elliptic_ic_train_shared.py) is the source code for training the iUzawa-Net with shared layer weights for solving the ill-conditioned elliptic optimal control problems in Section 6.2 of the [paper](https://arxiv.org/abs/2602.12273).
- [`parabolic_test.py`](https://github.com/tianyouzeng/iUzawa-Net/blob/main/parabolic_test.py) is the source code for testing the trained iUzawa-Net for the parabolic optimal control problems in Section 6.3 of the [paper](https://arxiv.org/abs/2602.12273).

Besides the files in the root directory:

- The [`data`](https://github.com/tianyouzeng/iUzawa-Net/tree/main/data) folder contains the code for generating training sets and testing sets.
- The [`fno`](https://github.com/tianyouzeng/iUzawa-Net/tree/main/fno) folder contains the code for training and testing the FNO models that we compared with in the paper.
- The [`trad_alg`](https://github.com/tianyouzeng/iUzawa-Net/tree/main/trad_alg) folder contains the implementaion of some traditional numerical algorithms that we compared with in the paper.
- The [`trained_models`](https://github.com/tianyouzeng/iUzawa-Net/tree/main/trained_models) folder contains some models trained by the code in this repository.
- The [`env`](https://github.com/tianyouzeng/iUzawa-Net/tree/main/data) folder contains an example [conda](https://docs.conda.io/en/latest) environment for running the code.

## Datasets

The generated training and testing datasets are not included in this repository due to GitHub's file size limitations. They can be found in this [OneDrive folder](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/logic_connect_hku_hk/IgCO2ltblrvJSZ8PFNx3dwr9AZX7RJ0-MSkZJD4CN1vJ4W4?e=lR0yZv).

## Citation

```
@misc{song2026learning,
      title={Learning to Control: The iUzawa-Net for Nonsmooth Optimal Control of Linear PDEs}, 
      author={Yongcun Song and Xiaoming Yuan and Hangrui Yue and Tianyou Zeng},
      year={2026},
      eprint={2602.12273},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2602.12273}, 
}
```