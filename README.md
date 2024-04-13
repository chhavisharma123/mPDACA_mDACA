This repository contains the implementation of algorithms developed in the following paper

**"Distributed Accelerated Gradient Methods with Restart under Quadratic Growth Condition". \
To Appear in Journal of Global Optimization, 2024.**
## Short Summary
This paper studies Nesterov Accelerated Gradient Method in the decentralized environment for solving constrained and unconstrained convex minimization problems with functions satisfying Quadratic Growth Condition (QGC). The algorithms designed in this work do not require the knowledge of QGC parameter, which is difficult to find in practice. The inclusion of restarting the update process and careful construction of local adaptive projection sets are proven to be useful in achieving the fast convergence of the proposed algorithms mPDACA (constrained setting) and mDACA (unconstrained setting).

## Code
**Constrained setting:** We have implemented mPDACA on regularized logistic regression over $\ell_1$ ball of radius 1 and $10^6$.
**Unconstrained setting:** Algorithm mDACA is implemented on the logistic regression problem.
