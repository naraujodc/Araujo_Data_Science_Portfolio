# Lotka-Volterra Predator-Prey Project
The purpose of this project is to explore the famous Lotka-Volterra predator-prey model and its applications to the analysis of population dynamics using R. I find population equilibria, generate phase-plane plots with null-clines, solve the model, make it more realistic, and investigate predator invasion.

**Skills:** Dynamical models, ordinary differential equations

![RStudio](https://img.shields.io/badge/RStudio-1f65cc?style=flat&logo=rstudioide&logoColor=%2375AADB)&nbsp;
![deSolve](https://img.shields.io/badge/deSolve-blue)&nbsp;

## Table of Contents
[1. The Lotka-Volterra Predatory-Prey Model](#the-lotka-volterra-predator-prey-model)\
[2. The Lotka-Volterra Model with a Carrying Capacity for Prey](#the-lotka-volterra-model-with-a-carrying-capacity-for-prey)\
[3. Predator Invasion](#predator-invasion)\
[4. Final Remarks](#final-remarks)\
[5. Files](#files)\
[6. Instructions for Use](#instructions-for-use)

## The Lotka-Volterra Predator-Prey Model
This model is a simple and famous framework to study the population dynamics of two species in a predator-prey relationship within an ecosystem. The simplest possible configuration of the predator-prey model consists of two first-order nonlinear differential equations:

$$
\frac{dn_1}{dt}=r n_1 - \alpha n_1 n_2
$$

$$
\frac{dn_2}{dt}=\epsilon \alpha\ n_1 n_2-\mu n_2
$$

Here, all parameters ($r,\alpha, \epsilon, \mu$) are greater than zero, and $\epsilon < 1$.

#### The variables:
- $\frac{dn_1}{dt}$ &rarr; The rate of change of the **prey** population (number of individuals per time unit)
- $n_1$ &rarr; The size of the **prey** population (number of individuals)
- $\frac{dn_2}{dt}$ &rarr; The rate of change of the **predator** population (number of individuals per time unit)
- $n_2$ &rarr; The size of the **predator** population (number of individuals)

#### The parameters:
- $r$ &rarr; The growth of the **prey** population through means other than predation (number of individuals per individual per time unit)
- $\alpha$ &rarr; Linear/Type I consumption rate of **prey** by **predator** &rarr; $\alpha n_1 n_2$ follows the Law of Mass-Action
  - Total rate of contact of predator and prey $*$ chance of predator successfully consuming prey
- $\epsilon$ &rarr; The conversion factor/rate by which the prey units are converted into predator
  - This corresponds to the fact that prey is converted into predator biomass, which predators use for reproduction
- $\mu$ &rarr; The rate of change of the **predator** population in the abscence of prey (number of individuals per individual per time unit)
  - Accounts for death, emigration, etc.

To explore predator-prey dynamics and later solve the system of differential equations, I created this function:
```
predator.prey = function(n1, n2, params) {
  r = params[1]
  alpha = params[2]
  epsilon = params[3]
  mu = params[4]
  
  dn1 = r * n1 - alpha * n1 * n2
  dn2 = epsilon * alpha * n1 * n2 - mu * n2
  
  return(c(dn1, dn2))
}
```

### Solving for equilibria
When both populations are in equilibrium, both of their rates of change are zero, i.e. $\frac{dn_1}{dt}=\frac{dn_2}{dt}=0$. By setting the two differential equations equal to zero, I found two possible equilibria for this model:
1. **Co-existence equilibrium**

$$
\hat{n}_1 = \frac{\mu}{\epsilon\alpha}
$$

$$
\hat{n}_2 = \frac{r}{\alpha}
$$

2. **"Double extinction" equilibrium**

$$
\hat{n}_1 = \hat{n}_2 = 0
$$

### Phase-Plane Plot
I plotted a phase-plane diagram to visualize the state of the predator population versus the state of the prey population. I used an arbitrary case where $r=0.1$, $\alpha=0.005$, $\epsilon=0.4$, and $\mu=0.08$. Here, the arrows indicate the direction of change of the populations.

<p align="center">
<img width=411 height=386.25 src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/phase%20arrows.png">
<p />

The curves I added to the diagram are the **null clines**, which indicate where one of the variables remains constant and correspond to the co-existence equilibrium equations I found previously. Their point of intersection represents the equilibrium for the whole system. In addition, it is possible to see that the null clines divide the phase-plane space into four sectors:

<p align="center">
<img width=603.5 height=347.25 src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/phase%20arrows%20labeled.jpeg">
<p />

In the phase-plane plot, the prey and predator populations seem to flow in a circle around the co-existence equilibrium. At some points, both of them increase/decrease at the same time, whereas at other points one of them increases while the other decreases, always cyclically. Thus, I predicted that the solution to this model would reveal a cyclical, wavelike pattern of increasing and decreasing populations, with the moments of increase/decrease partially overlapping.

### Solution to the Model
I used the ordinary differential equation (ODE) solver from the R `deSolve` library to find the solutions for the Lotka-Volterra predator-prey model. I used the same parameters as in the phase-plane plot ($r=0.1$, $\alpha=0.005$, $\epsilon=0.4$, and $\mu=0.08$) and the initial conditions $n_1=60$, $n_2=50$.

<p align="center">
<img width=308.25 height=289.6875 src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/n1%20ode.png">
<img width=308.25 height=289.6875  src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/n2%20ode.png">
<img width=308.25 height=289.6875  src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/n1%20n2%20ode.png">
<p />

The solution is consistent with my analysis of the phase-plane plot because it shows a cyclical pattern of increasing and decreasing populations, with these moments being sometimes synchronized and sometimes opposed. As it can be easily observed in the overlaid plot, $n_1$ goes from decreasing to increasing and vice-versa when $n_2=20$. Similarly, $n_2$ goes from increasing to decreasing and vice-versa when $n_1=40$. This reflects exactly the point of intersection of the null-clines in the phase-arrows plot.

## The Lotka-Volterra Model with a Carrying Capacity for Prey
To make the model more realistic, we can add a carrying capacity for the prey population. **Carrying capacity** is the maximum population size that an environment can sustain given its resources. 

$$
\frac{dn_1}{dt}=r n_1 \left(1- \frac{n_1}{K}\right)- \alpha n_1 n_2
$$

$$
\frac{dn_2}{dt}=\epsilon \alpha\ n_1 n_2-\mu n_2
$$

Here, all parameters ($r$, $\alpha$, $\epsilon$, $\mu$, $K$) are greater than zero.
- $K$ &rarr; Carrying capacity for prey population (number of individuals)

To explore predator-prey dynamics and later solve the system of differential equations, I created this function:
```
predator.prey.K = function(n1, n2, params) {
  r = params[1]
  alpha = params[2]
  epsilon = params[3]
  mu = params[4]
  K1 = params[5]
  
  dn1 = r * n1 * (1 - n1 / K1) - alpha * n1 * n2
  dn2 = epsilon * alpha * n1 * n2 - mu * n2
  
  return(c(dn1, dn2))
}
```

### Solving for Equilibria
Again, it is possible to derive two equilibria for this model: co-existence and "double extinction." I found the following null-clines for the co-existence equilibrium:

$$
n_1 = \frac{\mu}{\epsilon\alpha}
$$

$$
n_2 = \frac{r}{\alpha} - \frac{r}{K\alpha} n_1
$$

### Phase-Plane Plot
I plotted a phase-plane diagram to visualize the state of the predator population versus the state of the prey population. I used the same arbitrary case as before where $r=0.1$, $\alpha=0.005$, $\epsilon=0.4$, and $\mu=0.08$, with the additional parameter $K=50$. Here, the arrows indicate the direction of change of the populations.

<p align="center">
<img width=411 height=386.25 src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/phase%20arrows%20prey%20predator%20k1.png">
<p />

Once again, the null-clines divide the phase-plane plot into four sectors and intersect at the co-existence equilibrium:

<p align="center">
<img width=603.5 height=347.25 src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/phase%20arrows%20k1%20labeled.jpeg">
<p />

### Solution to the Model
I used the ordinary differential equation (ODE) solver from the R `deSolve` library to find the solutions for the Lotka-Volterra predator-prey model with a carrrying capacity for prey. I used the same parameters as in the phase-plane plot ($r=0.1$, $\alpha=0.005$, $\epsilon=0.4$, $\mu=0.08$, and $K=50$) and the initial conditions $n_1=60$, $n_2=50$.

<p align="center">
<img width=308.25 height=289.6875 src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/n1%20k1%20ode.png">
<img width=308.25 height=289.6875  src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/n2%20k1%20ode.png">
<img width=308.25 height=289.6875  src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/n1n2%20k1%20ode.png">
<p />

When a carrying capacity is added to the system, the solution of the ODE shows that, instead of following a cyclical pattern of oscillations, predator and prey stabilize at values corresponding to the co-existence equilibrium. These values can be seen at the intersection of the null-clines in the phase-space plot. Unlike in the first model I solved for, this is a stable equilibrium.

## Predator Invasion
Finally, I decided to investigate under what circumstances predator invasion is possible. In this scenario, the prey population starts at its carrying capacity, with no predators to disturb it. Then, one predator is introduced into the system.

To discover which parameter values would allow for successful invasion, I solved the model with the same parameters as before ($r=0.1$, $\alpha=0.005$, $\epsilon=0.4$, and $\mu=0.08$), but with varying values of $K$. In addition, I used the initial conditions $n_1=K$ and $n_2=1$.

<p align="center">
<img width=308.25 height=289.6875 src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/ode10.png">
<img width=308.25 height=289.6875  src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/ode30.png">
<img width=308.25 height=289.6875  src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/ode50.png">
<p />

<p align="center">
<img width=308.25 height=289.6875 src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/ode75.png">
<img width=308.25 height=289.6875  src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/ode100.png">
<img width=308.25 height=289.6875  src="https://github.com/naraujodc/Araujo_Data_Science_Portfolio/blob/main/Lotka_Volterra_Predator_Prey_Project/images/ode1000.png">
<p />

For a successful invasion, we must have a value of $K>\frac{\mu}{\epsilon\alpha}$, which is the co-existence equilibrium value for $n_1$. By looking at the phase-space plot, we can predict that any initial values $n_2=1$ and $n_1<\frac{\mu}{\epsilon\alpha}$, $n_1$ will tend to increase back to $\frac{\mu}{\epsilon\alpha}$ and $n_2$ will decrease to 0. This is what I observed when solving for $n_1=K=10$ and $n_1=K=30$, as pictured above. The predator went extinct. However, when $n_1=K>\frac{\mu}{\epsilon\alpha}$, there are initial oscillations in the populations of predator and prey, but they always stabilize at the equilibrium values found at the intersection of the null-clines in the phase-space plot. The larger $K$ is, the more oscillations happen before equilibrium is reached.

## Final Remarks
The Lotka-Volterra Model is very useful to study the interactions of predator-prey populations with a fairly simple computation. However, its simplicity implies assumptions that might be unrealistic, such as the fact that the prey population has access to unlimited resources. Adding a carrying capacity for the prey population allows us to make the model more realistic and simulate different scenarios such as a predator invasion.

## Files
- `Lotka_Volterra_Predator_Prey_.R` &rarr; This R script contains all the code used throughout this project to create and solve all the models and to generate the figures.
- `images` &rarr; This folder contains all the plots generated with the R script and distributed across this README.md file.
- `README.md` &rarr; Complete documentation for the Lotka-Volterra Predator-Prey Project.

## Instructions for Use
1. Download and open `Lotka_Volterra_Predator_Prey_.R`.
2. If you do not have the `deSolve` library installed, use `install.packages("deSolve")`.
3. Run the R script!
