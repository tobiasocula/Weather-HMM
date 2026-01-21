## Discrete case

Let $N$ be the amount of possible states, $M$ the amount of possible observations and $T$ the amount of timestamps we are considering.

$A\in\mathbb{R}^{N\times N}\\ ,B\in\mathbb{R}^{N\times M}\\ ,\pi\in\mathbb{R}^N$

where

$a_{ij}=$ transition probability of states $i\to j$

$b_{ik}=$ probability of obverving observation $k$ in state $i$

$\pi_i=$ probability of initially being in state $i$

### Iteration

We initialize $A$, $B$, $\pi$.

#### Expectation step

##### Forward algorithm

$\alpha\in\mathbb{R}^{T\times N}$

$\alpha_t(i)=\mathbb{P}(O_1,\dots,O_t\ |\ S_t=i)$

Compute:

$\alpha_1(i)=\pi_i b_i(O_{t=1})$

$\alpha_t(i)=b_i(O_{t=t})\sum_j\alpha_{t-1}(i)a_{ij}$

##### Backward algorithm

$\beta\in\mathbb{R}^{T\times N}$

$\beta_t(i)=\mathbb{P}(O_{t+1},\dots,O_T\ |\ S_t=i)$

Compute:

$\beta_T(i)=1$

$\beta_t(i)=\sum_j a_{ij}b_j(O_{t+1}\beta_{t+1}(j))$

$\gamma\in\mathbb{R}^{T\times N}$

$\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{\sum_j\alpha_t(j)\beta_t(j)}$

$\xi\in\mathbb{R}^{T\times N\times N}$

$\xi_t(i,j)=\frac{\alpha_t(i)a_{ij}b_j(O_{t+1})\beta_{t+1}(j)}{\sum_{k,l}\alpha_t(k)a_{kl}b_l(O_{t+1})\beta_{t+1}(l)}$

#### Maximalization step

$a_{ij}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}$

$b_{ik}=\frac{\sum_{t=1}^T\gamma_t(i)1_{O_t=k}}{\sum_{t=1}^T\gamma_t(i)}$

### Log likelyhood

$\text{log}\sum_i\alpha_T(i)$.

## Continuous emissions

We now define $M$ not as the amount of possible observation values for each state, but as the amount of "components" of each obervation (since now each obervation is continuous).

Now $B\in\mathbb{R}^{T\times N}$ is a time-dependent matrix, defined by $b_t(i)$ being the probability distribution of observation at time $t$ for state $i$:

$b_t(i)\sim\text{PDF}(x_t\ |\ \theta_i)$

where $\theta_i$ is the parameter set for the probability distribution for state $i$.

In general:

$\theta_{i}^{\text{new}}=\text{argmax}_{\theta_i}\sum_{t=1}^T\gamma_t(i)\text{log}(b_t(i))$