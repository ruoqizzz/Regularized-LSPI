# Reinforcement Learning: Linear-in-parameter model

## Reading

1. Statistical Reinforcement Learning: Chapter 2 - **Policy Iteration with Value Function Approximation**

   - Framework of *policy iteration* 

     - based on state value functions

     Policy evaluation: ![](http://latex.codecogs.com/gif.latex?V^{\pi}(s)=\mathbb{E}_{p(s'|s,a)\pi(a|s)}[r(s,a,s')+\gamma V^{\pi}(s')])
     
Policy improvement: ![](http://latex.codecogs.com/gif.latex?\pi^*(a|s)\leftarrow\delta(a-a^{\pi}(s)))
     
where ![](http://latex.codecogs.com/gif.latex?a^{\pi}(s)=argmax_{a\in\mathcal{A}}\{\mathbb{E}_{p(s'|s,a)\pi(a|s)}[r(s,a,s')+\gammaV^{\pi}(s')]\})
     
- based on state-action value function
     
Policy evaluation: ![](http://latex.codecogs.com/gif.latex?Q^{\pi}(s,a)=\mathbb{E}_{\pi(a'|s')p(s'|s,a)}[r(s,a)+\gammaQ^{\pi}(s',a')])
     
Policy improvement": ![](http://latex.codecogs.com/gif.latex?\pi^*(a|s)\leftarrow\delta(a-argmax_{a'\in\mathcal{A}}Q^\pi (s,a'))
     
- Value function approximation step to regression problem
   
- Least-squares policy iteration
   
  - Reward Regression and algorithm
   
    ![](http://latex.codecogs.com/gif.latex?\psi(s,a)=\phi(s,a)-\gamma\mathbb{E}_{\pi(a'|s')p(s'|s,a)}[\phi(s',a')])
   
    ![](http://latex.codecogs.com/gif.latex?r(s,a)\approx\theta^T\psi(s,a))
   
  - Regularization
   
2. **[Least-Squares Policy Iteration](http://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf)**

3. **[Towards Generalization and Simplicity in Continuous Control](http://papers.nips.cc/paper/7233-towards-generalization-and-simplicity-in-continuous-control)**

4. Reinforcement Learning: Charpter 9 - **On-policy Prediction with Approximation**

