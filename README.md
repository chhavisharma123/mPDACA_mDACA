This repository contains the implementation of algorithms developed in the following paper:

**"Distributed Accelerated Gradient Methods with Restart under Quadratic Growth Condition". \
Journal of Global Optimization, 2024. [pdf](https://rdcu.be/dFGsc)**
# Paper summary
This paper studies Nesterov Accelerated Gradient Method in the decentralized environment for solving constrained and unconstrained convex minimization problems with functions satisfying Quadratic Growth Condition (QGC). The algorithms designed in this work do not require the knowledge of QGC parameter, which is difficult to find in practice. The inclusion of restarting the update process and careful construction of local adaptive projection sets are proven to be useful in achieving the fast convergence of the proposed algorithms mPDACA (constrained setting) and mDACA (unconstrained setting).

# Code
**Constrained setting:** We have implemented mPDACA on regularized logistic regression over $\ell_1$ ball of radius 1 and $10^6$.

$$\min_{ \left\Vert x \right\Vert_1 \leq R } f(x) = \frac{1}{N}\sum_{i = 1}^{N} \log( 1+ \exp({-b_{i} x^\top a_{i}} ) + \frac{\lambda}{2}  \left\Vert x \right\Vert^2,$$\
where $\left\lbrace a_{i},b_{i}\right\rbrace_{i=1}^N$ is a binary classification dataset and $R$ is the radius of $\ell_1$ ball.

**Unconstrained setting:** Algorithm mDACA is implemented on the logistic regression problem

$$\min_{x \in \mathbb{R}^{d}} \frac{1}{N}\sum_{i = 1}^{N} \log( 1+ \exp({-b_{i}  x^\top a_{i}} )).$$

We use seven datasets from [LIBSVM repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) and [Causality workbench](https://www.causality.inf.ethz.ch/data/SIDO.html). The codes contained in this repo can be implemented for different datasets by changing the file name in data loading cell of the jupyter notebook.

### Run
#### Dependencies 
```bash
python==3.7
```
```bash
Packages required: numpy, random, networkx, math, matplotlib.pyplot, complete_bipartite_graph, linalg, csv, sklearn.utils
```
#### Files
 1. mPDACA_a4a_2D_Torus.ipynb(.py) and dataset are available inside constrained-setting directory. Corresponding files for unconstrained setting are available inside unconstrained-setting directory.
3. Output: function values, proximal gradient norm values and number of communications are saved to .txt files.

# Citation
```bash
TBA
```
# Contact
For any query, please do reach out at
```bash
chhavisharma760@gmail.com
fchhavi@smu.edu
```
