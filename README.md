# BPRH

This repository implements the model from [Qiu, Huihuai, et al. "BPRH: Bayesian personalized ranking for heterogeneous implicit feedback" Information Sciences 453 (2018): 80-98](https://www.sciencedirect.com/science/article/pii/S0020025516315742).

## Platform and Packages

The codes are programmed and tested on python 3.7.6. And they should also run on other versions of python.

[bprH.py](https://github.com/liu-yihong/BPRH/blob/master/bprH.py) is the basic model wrapped in class for convenient usage. Packages below are required to run [bprH.py](https://github.com/liu-yihong/BPRH/blob/master/bprH.py)
- pickle
- random
- numpy
- pandas
- tqdm
- livelossplot
- scikit-learn

Since repeated vector and matrix manipulations are involved in BPRH model. [bprH_gpu.py](https://github.com/liu-yihong/BPRH/blob/master/bprH_gpu.py)  leverage the power of NVIDIA GPU for acceleration. Package [CuPy](https://cupy.dev/) is required to run [bprH_gpu.py](https://github.com/liu-yihong/BPRH/blob/master/bprH_gpu.py). You may check [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html) for installment guidance.

## Implementation Detail

## Mathematical Detail

For mathematical details, please visit [my blogs](https://liu-yihong.github.io/2020/06/26/Understanding-BPR-COFISET-and-BPRH/).

Please cite this repository if you use my codes.
