# clearing past actions from memory
rm(list=ls())

## lotka-volterra predator-prey model

# creating predator-prey function
predator.prey = function(n1, n2, params) {
  r = params[1]
  alpha = params[2]
  epsilon = params[3]
  mu = params[4]
  
  dn1 = r * n1 - alpha * n1 * n2
  dn2 = epsilon * alpha * n1 * n2 - mu * n2
  
  return(c(dn1, dn2))
}

# creating phase arrows function
phasearrows = function(fun,xlims,ylims,resol=25, col='black',
                       add=F,parms=NULL,jitter=FALSE) {
  if (add==F) {
    plot(1,xlim=xlims, ylim=ylims, type='n',xlab="n1",ylab="n2")
  }
  x = matrix(seq(xlims[1],xlims[2], length=resol), byrow=T, resol,resol)
  y = matrix(seq(ylims[1],ylims[2], length=resol),byrow=F, resol, resol)
  npts = resol*resol
  z = fun(x,y,parms)
  z1 = matrix(z[1:npts], resol, resol)
  z2 = matrix(z[(npts+1):(2*npts)], resol, resol)
  maxx = max(abs(z1))
  maxy = max(abs(z2))
  dt = min( abs(diff(xlims))/maxx, abs(diff(ylims))/maxy)/resol
  lens = sqrt(z1^2 + z2^2)
  lens2 = lens/max(lens) 
  arrows(c(x), c(y), c(x+dt*z1/((lens2)+.1)), c(y+dt*z2/((lens2)+.1)),
         length=.04, col=col)
}

# create phase arrows plot
phasearrows(predator.prey, xlims = c(0,100), ylims = c(0,100),
            parms = c(0.1, 0.005, 0.4, 0.08))

# add null clines, color for easier visualization
abline(h = 0.1 / 0.005, col = 2, lwd = 2) # for dn1 = 0
abline(v = 0.08 / (0.4 * 0.005), lty = 2, col = 2, lwd = 2) # for dn2 = 0

## solving the model

# modified predator-prey model ordinary differential equation
predator.prey.ode = function(curr.time, myvars, params) {
  r = params[1]
  alpha = params[2]
  epsilon = params[3]
  mu = params[4]
  
  n1 = myvars[1]
  n2 = myvars[2]
  
  dn1 = r * n1 - alpha * n1 * n2
  dn2 = epsilon * alpha * n1 * n2 - mu * n2
  
  return(list(c(dn1, dn2)))
}

# load desolve libraby
library(deSolve)

# solve differential equation
out = ode(y = c(60, 50), times = c(1:5000)*0.1,
          func = predator.prey.ode, parms = c(0.1, 0.005, 0.4, 0.08))

# plot n1(t)
plot(out[,1], out[,2], type = "l", xlab = "Time", ylab = "n1")

# plot n2(t)
plot(out[,1], out[,3], type = "l", xlab = "Time", ylab = "n2")

# plot n1(t) and n2(t) in the same plot
plot(out[,1], out[,2], type = "l", ylim = c(0,150), xlab = "Time", ylab = "n")
lines(out[,1], out[,3], col = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)


## predator-prey model with carrying capacity for the prey

# creating predator-prey function with carrying capacity
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

# create phase arrows plot
phasearrows(predator.prey.K, xlims = c(0,100), ylims = c(0,100),
            parms = c(0.1, 0.005, 0.4, 0.08, 50), resol = 23)

# add null clines, color for easier visualization
abline(0.1 / 0.005, (-0.1 / (50 * 0.005)), col = 2, lwd = 2) # for dn1 = 0
abline(v = 0.08 / (0.4 * 0.005), lty = 2, col = 2, lwd = 2) # for dn2 = 0

# modified predator-prey w/ carrying capacity ODE function
predator.prey.K.ode = function(curr.time, myvars, params) {
  r = params[1]
  alpha = params[2]
  epsilon = params[3]
  mu = params[4]
  K1 = params[5]
  
  n1 = myvars[1]
  n2 = myvars[2]
  
  dn1 = r * n1 * (1 - n1 / K1) - alpha * n1 * n2
  dn2 = epsilon * alpha * n1 * n2 - mu * n2
  
  return(list(c(dn1, dn2)))
}

# solve differential equation
out.K = ode(y = c(60, 50), times = c(1:5000)*0.1,
            func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 50))

# plot n1(t)
plot(out.K[,1], out.K[,2], type = "l", xlab = "Time", ylab = "n1")

# plot n2(t)
plot(out.K[,1], out.K[,3], type = "l", xlab = "Time", ylab = "n2")

# plot n1(t) and n2(t) in the same plot
plot(out.K[,1], out.K[,2], type = "l", ylim = c(0,80),
     xlab = "Time", ylab = "n")
lines(out.K[,1], out.K[,3], col = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)


## finding circumstances for invasion

# testing with initial conditions n1 = K1 = 50 and n2 = 1
out.50 = ode(y = c(50, 1), times = c(1:5000)*0.1,
            func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 50))

plot(out.50[,1], out.50[,2], type = "l", ylim = c(0,80),
     xlab = "Time", ylab = "n")
lines(out.50[,1], out.50[,3], col = 2)
abline(h = 50, lty = 2)
abline(h = 0, lty = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)
title("Predator invasion with K1 = 50")

# testing with initial conditions n1 = K1 = 25 and n2 = 1
out.25 = ode(y = c(25, 1), times = c(1:5000)*0.1,
             func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 25))

plot(out.25[,1], out.25[,2], type = "l", ylim = c(0,40),
     xlab = "Time", ylab = "n")
lines(out.25[,1], out.25[,3], col = 2)
abline(h = 25, lty = 2)
abline(h = 0, lty = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)
title("Predator invasion with K1 = 25")

# testing with initial conditions n1 = K1 = 75 and n2 = 1
out.75 = ode(y = c(75, 1), times = c(1:5000)*0.1,
             func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 75))

plot(out.75[,1], out.75[,2], type = "l", ylim = c(0,100),
     xlab = "Time", ylab = "n")
lines(out.75[,1], out.75[,3], col = 2)
abline(h = 75, lty = 2)
abline(h = 0, lty = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)
title("Predator invasion with K1 = 75")

# testing with initial conditions n1 = K1 = 100 and n2 = 1
out.100 = ode(y = c(100, 1), times = c(1:5000)*0.1,
             func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 100))

plot(out.100[,1], out.100[,2], type = "l", ylim = c(0,120),
     xlab = "Time", ylab = "n")
lines(out.100[,1], out.100[,3], col = 2)
abline(h = 100, lty = 2)
abline(h = 0, lty = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)
title("Predator invasion with K1 = 100")

# testing with initial conditions n1 = K1 = 10 and n2 = 1
out.10 = ode(y = c(10, 1), times = c(1:5000)*0.1,
             func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 10))

plot(out.10[,1], out.10[,2], type = "l", ylim = c(0,15),
     xlab = "Time", ylab = "n")
lines(out.10[,1], out.10[,3], col = 2)
abline(h = 10, lty = 2)
abline(h = 0, lty = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)
title("Predator invasion with K1 = 10")

# testing with initial conditions n1 = K1 = 1000 and n2 = 1
out.1000 = ode(y = c(1000, 1), times = c(1:50000)*0.1,
              func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 1000))

plot(out.1000[,1], out.1000[,2], type = "l", ylim = c(0,1200),
     xlab = "Time", ylab = "n")
lines(out.1000[,1], out.1000[,3], col = 2)
abline(h = 1000, lty = 2)
abline(h = 0, lty = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)
title("Predator invasion with K1 = 1000")

# testing with initial conditions n1 = K1 = 30 and n2 = 1
out.30 = ode(y = c(30, 1), times = c(1:5000)*0.1,
             func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 30))

plot(out.30[,1], out.30[,2], type = "l", ylim = c(0,40),
     xlab = "Time", ylab = "n")
lines(out.30[,1], out.30[,3], col = 2)
abline(h = 30, lty = 2)
abline(h = 0, lty = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)
title("Predator invasion with K1 = 30")

# testing with initial conditions n1 = K1 = 40 and n2 = 1
out.40 = ode(y = c(40, 1), times = c(1:5000)*0.1,
             func = predator.prey.K.ode, parms = c(0.1, 0.005, 0.4, 0.08, 40))

plot(out.40[,1], out.40[,2], type = "l", ylim = c(0,50),
     xlab = "Time", ylab = "n")
lines(out.40[,1], out.40[,3], col = 2)
abline(h = 40, lty = 2)
abline(h = 0, lty = 2)
legend("topleft", c("n1", "n2"), col = c(1,2), lwd = 1)
title("Predator invasion with K1 = 40")