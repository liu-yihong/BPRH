# BPRH

This repository implements the model from [Qiu, Huihuai, et al. "BPRH: Bayesian personalized ranking for heterogeneous implicit feedback" Information Sciences 453 (2018): 80-98](https://www.sciencedirect.com/science/article/pii/S0020025516315742).

## Platform and Packages

The codes are programmed and tested on python 3.7.6. And they should also run on other versions of python.

[bprH.py](https://github.com/liu-yihong/BPRH/blob/master/bprH.py) is the basic model wrapped in class for convenient usage. Packages below are required to run [bprH.py](https://github.com/liu-yihong/BPRH/blob/master/bprH.py)
- pickle
- random
- numpy==1.18.1
- pandas==1.0.1
- tqdm==4.42.1
- livelossplot==0.5.1
- scikit-learn==0.22.1

Since repeated vector and matrix manipulations are involved in BPRH model. [bprH_gpu.py](https://github.com/liu-yihong/BPRH/blob/master/bprH_gpu.py)  leverage the power of NVIDIA GPU for acceleration. Package [CuPy](https://cupy.dev/) is required to run [bprH_gpu.py](https://github.com/liu-yihong/BPRH/blob/master/bprH_gpu.py). You may check [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html) for installation help. The version we used is cupy-cuda101==7.3.0 and CUDA 10.1.

[Sobazaar_cleaning.ipynb](https://github.com/liu-yihong/BPRH/blob/master/Sobazaar_cleaning.ipynb) is the Jupyter Notebook that cleans the raw Sobazaar data "[Sobazaar-hashID.csv.gz](https://github.com/liu-yihong/BPRH/blob/master/data/Sobazaar-hashID.csv.gz)" located in [data](https://github.com/liu-yihong/BPRH/tree/master/data) folder. You may unzip it manually before execute [Sobazaar_cleaning.ipynb](https://github.com/liu-yihong/BPRH/blob/master/Sobazaar_cleaning.ipynb). Notice that we do not consider Like action and only View action will get processed in [bprH_gpu.py](https://github.com/liu-yihong/BPRH/blob/master/bprH_gpu.py) and [bprH.py](https://github.com/liu-yihong/BPRH/blob/master/bprH.py). 

[BRPH_50_1000_0.00001_0.1_0.1.ipynb](https://github.com/liu-yihong/BPRH/blob/master/BRPH_50_1000_0.00001_0.1_0.1.ipynb) illustrate the usage and training process of BPRH on GPU.

## Parameters Sensitivity Analysis

| gamma | lambda_u, lambda_v | lambda_b |  P@5  |  P@10 |  R@5  |  R@10 |  AUC  |
| :-------: | :-----------------------: | :----------: | :-----: | :-----: | :-----: | :-----: | :-----: |
|    0.1    |        0.00001         |  0.00001  | 0.014 | 0.011 | 0.061 | 0.091 | 0.857 |
|    0.1    |        0.00001         |   0.0001   | 0.014 | 0.011 | 0.062 | 0.094 | 0.858 |
|    0.1    |        0.00001         |   0.001    | 0.018 | 0.013 | 0.075 | 0.105 | 0.861 |
|    0.1    |        0.00001         |    0.01     | 0.033 | 0.021 | 0.146 | 0.175 | 0.866 |
|    0.1    |        0.00001         |     0.1     | 0.054 | 0.034 | 0.227 | 0.281 | 0.885 |
|    0.1    |         0.0001         |  0.00001  | 0.014 | 0.011 |  0.06 | 0.092 |  0.86 |
|    0.1    |         0.0001         |   0.0001   | 0.015 | 0.011 | 0.064 | 0.091 | 0.856 |
|    0.1    |         0.0001         |   0.001    | 0.016 | 0.012 | 0.071 | 0.106 |  0.86 |
|    0.1    |          0.001          |  0.00001  | 0.013 |  0.01  | 0.054 | 0.087 | 0.858 |
|    0.1    |          0.001          |   0.0001   | 0.014 | 0.011 | 0.058 | 0.089 | 0.859 |
|    0.1    |          0.001          |   0.001    | 0.016 | 0.011 | 0.069 | 0.097 | 0.859 |

We set the number of iterations as 720,000 for the table above. <img src="https://render.githubusercontent.com/render/math?math=\gamma = 0.1, \lambda_{u} = \lambda_{v} = 0.00001, \lambda_{b} = 0.1"> is selected for a 5-folds cross validation on 600,000 iterations. Results are presented belows.

| FOLD NUM |     P@5     |     P@10    |     R@5     |     R@10    |     AUC     |
|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|     0    | 0.051182432 | 0.032685811 | 0.198458486 | 0.247398244 | 0.872067812 |
|     1    | 0.048061105 | 0.031257344 | 0.198704581 | 0.253552421 | 0.869966503 |
|     2    | 0.048014226 | 0.031298162 | 0.204597253 | 0.259258681 | 0.870846781 |
|     3    | 0.045968712 | 0.030024067 | 0.195642696 | 0.250112863 | 0.863704698 |
|     4    | 0.046420048 | 0.031264916 | 0.191670788 | 0.253109041 | 0.874987955 |
|    AVG   | 0.047929305 |  0.03130606 | 0.197814761 |  0.25268625 |  0.87031475 |
|    STD   | 0.002045067 | 0.000942251 | 0.004734835 | 0.004435966 | 0.004153589 |

## Implementation Detail

This section includes the implementation details unmetioned in [Qiu, Huihuai, et al. "BPRH: Bayesian personalized ranking for heterogeneous implicit feedback" Information Sciences 453 (2018): 80-98](https://www.sciencedirect.com/science/article/pii/S0020025516315742).

1. There are nine types of action in the original Sobazaar dataset. We group 'purchase:buy_clicked' as Purchase, 'content:interact:product_clicked', 'content:interact:product_detail_viewed', 'product_detail_clicked' as View, and 'content:interact:product_wanted', 'product_wanted' as Like. Then we can get 4712 users and 7015 items with 15208 purchases, 126846 views, and 96689 likes. This is aligned to Table 4 in [Qiu, Huihuai, et al. "BPRH: Bayesian personalized ranking for heterogeneous implicit feedback" Information Sciences 453 (2018): 80-98](https://www.sciencedirect.com/science/article/pii/S0020025516315742).
2. For [auxiliary and target actions correlation](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L108), we only consider the case of View with Purchase. Hence, <img src="https://render.githubusercontent.com/render/math?math=\rho = 1">. What's more, on Sobazaar dataset, it is possible that <img src="https://render.githubusercontent.com/render/math?math=I_{a}^{u} \cap I_{t}^{u} = \emptyset">, leading to the 0-devided-by-0 error when calculating <img src="https://render.githubusercontent.com/render/math?math=C^{u}_{ta}, C^{u}_{at}, C^{u}">. Therefore, we set <img src="https://render.githubusercontent.com/render/math?math=\alpha_{u} = 1"> in this case.
3. For [item-set coselection](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L151), when item <img src="https://render.githubusercontent.com/render/math?math=i"> is only purchased by one user, then according to the definition of <img src="https://render.githubusercontent.com/render/math?math=S^{i} = \{ j | |U^{i} \cap U^{j}| \geq 2, i,j \in I\}"> is an empty set since <img src="https://render.githubusercontent.com/render/math?math=|U^{i}| = 1">. However, <img src="https://render.githubusercontent.com/render/math?math=S^{i}"> should contain item <img src="https://render.githubusercontent.com/render/math?math=i"> no matter what the size of <img src="https://render.githubusercontent.com/render/math?math=S^{i}"> is accroding to the paper. We fix this issue in our code.
4. For item-set coselection involved in Algorithm 1 in BPRH paper, we think there are some typos. Taking Line 20 - 21 as an instance, to construct the item-set <img src="https://render.githubusercontent.com/render/math?math=K">, first we randomly selection item <img src="https://render.githubusercontent.com/render/math?math=k \in I_{n}^{u}">, then <img src="https://render.githubusercontent.com/render/math?math=K "> should come from <img src="https://render.githubusercontent.com/render/math?math=K = I_{n}^{u} \cap S^{k}">, not <img src="https://render.githubusercontent.com/render/math?math=K = I_{n}^{u} \cap S^{i}">. So is the case of item-set <img src="https://render.githubusercontent.com/render/math?math=J">. We fix this issue in [bprH_gpu.py - Line 286](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L286), [Line 301](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L301), [Line 317](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L317).
5. BPRH model does not consider user bias. So we add a all-ones-column at the last column in user matrix and set the last row of item matrix as item bias [(bprH_gpu.py - Line 255)](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L255). We utilize normal distribution with 0 expectation and 0.1 standard deviation to initialize user and item matrices.
6. When constructing item-sets <img src="https://render.githubusercontent.com/render/math?math=I, J, K">, we may come across some empty item-set because of random spliting train and test dataset. [bprH_gpu.py - Line 363](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L363) address this issue. For example, when <img src="https://render.githubusercontent.com/render/math?math=J = \emptyset">, the objective function of BPRH and corresponding gradients downgrade to COFISET model.
7. When recommending items for users, a user might appear in test and not in train. In our implementation, we can choose to ignore this type of user, i.e. we do not recommend for this type of user. In another option, we use item popularity learned from training data to make recommendations for this type of users. [bprH_gpu.py - Line 520](https://github.com/liu-yihong/BPRH/blob/f8f74de1bd97783f7d5274d05096cbfc96fc0136/bprH_gpu.py#L520) solve this issue. What's more, we exclude user <img src="https://render.githubusercontent.com/render/math?math=u">'s purchased items from user <img src="https://render.githubusercontent.com/render/math?math=u">'s recommendation lists.


## Mathematical Detail

For mathematical details, please visit [my blogs](https://liu-yihong.github.io/2020/06/26/Understanding-BPR-COFISET-and-BPRH/).

## Copyright

This repository is under [MIT License](https://github.com/liu-yihong/BPRH/blob/master/LICENSE). Please cite this repository if you use our codes.
