# Saddle-Point ADMM (SP-ADMM)

Saddle-point alternating direction method of multipliers (SP-ADMM) is an algorithm to solve decomposable saddle point problems.
The decomposable saddle-point problem has the following form:

$$\begin{matrix}\underset{x_{a}}{\min} \ \underset{x_{b}}{\max} &  \sum_{1\leq i \leq N} f_{i}(x_{a,i}, x_{b,i}) & \\ 
\text{subject to} & x_{a, i} \in X_{a, i} & \text{for all } i \\ 
 & x_{b, i} \in X_{b, i} & \text{for all } i  \\
  & x_{a} \in X_{a} & \\
 & x_{b} \in X_{b}
  \end{matrix}$$

 where $x_{a} = [x_{a,1}, \ldots, x_{a,N}]$ and $x_{b} = [x_{b,1}, \ldots, x_{b,N}]$.

## Solving decomposable saddle-point problems with SP-ADMM
SP-ADMM is implemented through the SaddlePointADMM class located at src>saddle_point_admm.py file.
An instance of SP-ADMM is created by inheritence. 
Every class that inherits SaddlePointADMM requires the following functions:
- An initialization function
```
initialize_vars()
```
This function needs to return $x_{a}, x_{b}, z_{a}, z_{b}, \lambda_{a}, \lambda_{b}$ such that $z_{a} \in X_{a}$ and $z_{b} \in X_{b}$. 

- A saddle-point solver for the individual games
```
solve_augmented_saddle_game(z_a, z_b, lmd_a, lmd_b)
```
This function needs solve the saddle-point problem $$\underset{x_{a,i} \in X_{a, i} }{\min} \underset{x_{b} \in X_{b, i}}{\max} f_{i}(x_{a,i}, x_{b,i}) + \lambda_{a,i}(x_{a,i} - z_{a,i})+ \frac{\rho_{a}}{2}\lVert x_{a,i} - z_{a,i}\rVert_{2}^{2}-\lambda_{b,i}(x_{b,i} - z_{b,i}) - \frac{\rho_{b}}{2}\lVert x_{b,i} - z_{b,i}\rVert_{2}^{2}.$$

- Projection functions
```
project_z_a(vec)
project_z_b(vec)
```
These functions need solve the following projection problems
$$\underset{z_{a} \in X_{a} }{\min} \  \frac{\rho_{a}}{2}\lVert z_{a} - vec\rVert_{2}^{2}$$
and
$$\underset{z_{b} \in X_{b} }{\min} \  \frac{\rho_{b}}{2}\lVert z_{b} - vec\rVert_{2}^{2},$$

respectively.

## Getting started with examples
SP-ADMM is implemented on four different examples located in the folder 'examples'. To run these examples import the relevant script from main. These examples are
1) Weighted inner product game in a unit ball

$$\begin{matrix}
\underset{x_{a}}{\min} \ \underset{x_{b}}{\max} &  \sum_{1\leq i \leq N} c_{i} x_{a,i} x_{b,i}& \\ 
\text{subject to} & x_{a, i} \in [l_{a,i}, u_{a,i}] & \text{for all } i \\ 
 & x_{b, i} \in [l_{b,i}, u_{b,i}] & \text{for all } i  \\
 & \lVert {x_{a}} \rVert_{2}^{2} \leq 1 & \\
 & \lVert {x_{b}} \rVert_{2}^{2} \leq 1 & 
\end{matrix}$$

where 
   - $x_{a,i}$ and $x_{b,i}$ are one-dimensional, 
   - $c_{i}$, $l_{a,i}$, $l_{b,i}$, $u_{a,i}$, and $u_{b,i}$ are given constants.
    
This problem type is implemented with class InnerProductGameBall located at src>saddle_point_games>inner_product_game.
An example of this class is implemented in src>examples>inner_product_game_ball_example.py file.
   
2) Weighted inner product game on unit simplex

$$
\begin{matrix}
\underset{x_{a}}{\min} \ \underset{x_{b}}{\max} &  \sum_{1\leq i \leq N} c_{i} x_{a,i} x_{b,i}& \\ 
\text{subject to} & x_{a, i} \in [l_{a,i}, u_{a,i}] & \text{for all } i \\ 
 & x_{b, i} \in [l_{b,i}, u_{b,i}] & \text{for all } i  \\
 &  x_{a} \in \Delta_{N} & \\
 &  x_{b} \in \Delta_{N} & 
\end{matrix}
$$ 

where 
   - $x_{a,i}$ and $x_{b,i}$ are one-dimensional, 
   - $\Delta_{N}$ is the $N$-dimensional unit simplex, and
   - $c_{i}$, $l_{a,i}$, $l_{b,i}$, $u_{a,i}$, and $u_{b,i}$ are given constants.

This problem type is implemented with class InnerProductGameSimplex located at src>saddle_point_games>inner_product_simplex.
The solutions for the individual augmented saddle-point games are computed analytically using the first-order optimality conditions.
The projection functions are also use analytical solutions.
An example of this class is implemented in src>examples>inner_product_game_simplex_example.py file.
  
   

3) Power allocation game for communication channels

$$
\begin{matrix}
\underset{x_{a}}{\min} \ \underset{x_{b}}{\max} &  \sum_{1 \leq i \leq N} \log\left(1 + \frac{\beta x_{b,i}}{\sigma_{i} + x_{a,i}} \right)& \\ 
\text{subject to} & x_{a, i} \in [l_{a,i}, u_{a,i}] & \text{for all } i \\ 
 & x_{b, i} \in [l_{b,i}, u_{b,i}] & \text{for all } i  \\
 & x_{a}/A \in \Delta_{N} &  & \\
 & x_{b}/B \in \Delta_{N} &   & 
\end{matrix}
$$

where 
- $x_{a,i}$ and $x_{b,i}$ are one-dimensional, 
- $\Delta_{N}$ is the $N$-dimensional unit simplex, and
- $A$, $B$, $l_{a,i}$, $l_{b,i}$, $u_{a,i}$, and $u_{b,i}$ are given constants.

This problem type is implemented with class PowerAllocationGame located at src>saddle_point_games>power_allocation_game.
The solutions for the individual augmented saddle-point games are computed saddle-point Frank-Wolfe algorithm (See https://arxiv.org/pdf/1610.07797.pdf for details.)
The projection functions use bisection algorithm.    
An example of this class is implemented in src>examples>power_allocation_example.py file.


4) Network routing game with adversarial costs

$$
\begin{matrix}
\underset{x_{a}}{\min} \ \underset{x_{b}}{\max} &  \sum_{1 \leq i \leq N}  x_{a,i}(x_{a,i} + x_{b,i})& \\ 
\text{subject to} & x_{a, i} \in [0,1] & \text{for all } i \\ 
 & x_{b, i} \in [0,1] & \text{for all } i  \\
 & \sum_{i \in E} x_{a, i} \geq 0.1 &  &\\
 & x_{a} \in MDP_{X} &  & \\
 & x_{b} \in MDP_{X} &   & 
\end{matrix}
$$

where 
- $x_{a,i}$ and $x_{b,i}$ are one-dimensional that represent the density vectors of the player in the network, 
- $MDP_{X}$ is the set of valid density vectors for the MDP that represents the network, and
- $E$ is the incoming edges of state \(1\).

This problem type is implemented with class NetworkRoutingGame located at src>saddle_point_games>network_routing_game.
The solutions for the individual augmented saddle-point games are computed analytically using the first-order optimality conditions.
The projection functions solve an linear program.    
An example of this class is implemented in src>examples>power_allocation_example.py file.




