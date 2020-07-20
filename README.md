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

Since repeated vector and matrix manipulations are involved in BPRH model. [bprH_gpu.py](https://github.com/liu-yihong/BPRH/blob/master/bprH_gpu.py)  leverage the power of NVIDIA GPU for acceleration. Package [CuPy](https://cupy.dev/) is required to run [bprH_gpu.py](https://github.com/liu-yihong/BPRH/blob/master/bprH_gpu.py). You may check [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html) for installation guidance.

[Sobazaar_cleaning.ipynb](https://github.com/liu-yihong/BPRH/blob/master/Sobazaar_cleaning.ipynb) is the Jupyter Notebook that cleans the raw Sobazaar data "[Sobazaar-hashID.csv.gz](https://github.com/liu-yihong/BPRH/blob/master/data/Sobazaar-hashID.csv.gz)" located in [data](https://github.com/liu-yihong/BPRH/tree/master/data) folder. You may unzip it manually before execute [Sobazaar_cleaning.ipynb](https://github.com/liu-yihong/BPRH/blob/master/Sobazaar_cleaning.ipynb). Notice that we do not consider Like action and only View action will get processed in [bprH_gpu.py](https://github.com/liu-yihong/BPRH/blob/master/bprH_gpu.py) and [bprH.py](https://github.com/liu-yihong/BPRH/blob/master/bprH.py). 

[BRPH_test_0718_filtered5.ipynb](https://github.com/liu-yihong/BPRH/blob/master/BRPH_test_0718_filtered5.ipynb) illustrate the usage and training process of BPRH on GPU. In this experiment, we only focus on users who purchased more than 5 items.

## Implementation Detail

This section includes the implementation details unmetioned in [Qiu, Huihuai, et al. "BPRH: Bayesian personalized ranking for heterogeneous implicit feedback" Information Sciences 453 (2018): 80-98](https://www.sciencedirect.com/science/article/pii/S0020025516315742).

1. There are nine types of action in the original Sobazaar dataset. We group 'purchase:buy_clicked' as Purchase, 'content:interact:product_clicked', 'content:interact:product_detail_viewed', 'product_detail_clicked' as View, and 'content:interact:product_wanted', 'product_wanted' as Like. Then we can get 4712 users and 7015 items with 15208 purchases, 126846 views, and 96689 likes. This is aligned to Table 4 in [Qiu, Huihuai, et al. "BPRH: Bayesian personalized ranking for heterogeneous implicit feedback" Information Sciences 453 (2018): 80-98](https://www.sciencedirect.com/science/article/pii/S0020025516315742).
2. For [auxiliary and target actions correlation](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L108), we only consider the case of View with Purchase. Hence, <img src="https://render.githubusercontent.com/render/math?math=\rho = 1">. What's more, on Sobazaar dataset, it is possible that <img src="https://render.githubusercontent.com/render/math?math=I_{a}^{u} \cap I_{t}^{u} = \emptyset">, leading to the 0-devided-by-0 error when calculating <img src="https://render.githubusercontent.com/render/math?math=C^{u}_{ta}, C^{u}_{at}, C^{u}">. Therefore, we set <img src="https://render.githubusercontent.com/render/math?math=\alpha_{u} = 1"> in this case.
3. For [item-set coselection](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L151), when item <img src="https://render.githubusercontent.com/render/math?math=i"> is only purchased by one user, then according to the definition of <img src="https://render.githubusercontent.com/render/math?math=S^{i} = \{ j | |U^{i} \cap U^{j}| \geq 2, i,j \in I\}"> is an empty set since <img src="https://render.githubusercontent.com/render/math?math=|U^{i}| = 1">. However, <img src="https://render.githubusercontent.com/render/math?math=S^{i}"> should contain item <img src="https://render.githubusercontent.com/render/math?math=i"> no matter what the size of <img src="https://render.githubusercontent.com/render/math?math=S^{i}"> is accroding to the paper. We fix this issue in our code.
4. For item-set coselection involved in Algorithm 1 in BPRH paper, we think there are some typos. Taking Line 20 - 21 as an instance, to construct the item-set <img src="https://render.githubusercontent.com/render/math?math=K">, first we randomly selection item <img src="https://render.githubusercontent.com/render/math?math=k \in I_{n}^{u}">, then <img src="https://render.githubusercontent.com/render/math?math=K "> should come from <img src="https://render.githubusercontent.com/render/math?math=K = I_{n}^{u} \cap S^{k}">, not <img src="https://render.githubusercontent.com/render/math?math=K = I_{n}^{u} \cap S^{i}">. So is the case of item-set <img src="https://render.githubusercontent.com/render/math?math=J">. We fix this issue in [bprH_gpu.py - Line 286](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L286), [Line 301](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L301), [Line 317](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L317).


## Mathematical Detail

For mathematical details, please visit [my blogs](https://liu-yihong.github.io/2020/06/26/Understanding-BPR-COFISET-and-BPRH/).

## Copyright

This repository is under [MIT License](https://github.com/liu-yihong/BPRH/blob/master/LICENSE). Please cite this repository if you use my codes.
