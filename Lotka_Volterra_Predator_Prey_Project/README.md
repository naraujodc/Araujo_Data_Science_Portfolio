# Lotka-Volterra Predator-Prey Project
The purpose of this project is to explore the famous Lotka-Volterra predator-prey model and its applications to the analysis of population dynamics using R.

# The Lotka-Volterra Predator-Prey Model
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

2. **Double extinction equilibrium**

$$
\hat{n}_1 = \hat{n}_2 = 0
$$

### Phase-Plane Plot
I plotted a phase-plane diagram to visualize the state of the predator population versus the state of the prey population. Here, the arrows indicate the direction of change of the populations.

The curves I added to the diagram are the **null clines**, which indicate where one of the variables remains constant and correspond to the co-existence equilibrium equations I found previously. Their point of intersection represents the equilibrium for the whole system. In addition, it is possible to see that the null clines divide the phase-plane space into four sectors:

In the phase-plane plot, the prey and predator populations seem to flow in a circle around the co-existence equilibrium. At some points, both of them increase/decrease at the same time, whereas at other points one of them increases while the other decreases, always cyclically. Thus, I predicted that the solution to this model would reveal a cyclical, wavelike pattern of increasing and decreasing populations, with the moments of increase/decrease partially overlapping.
