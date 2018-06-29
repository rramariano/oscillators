"""
Simulation of different kinds of oscillations
"""

import numpy as np
import matplotlib.pyplot as plt

# Different kind of oscillators as functions
def simpleHarmonic(t,theta,omega):
    """
    with small angle approximation
    """
    g = 9.8
    l = g
    F = -(g*theta)/l
    return F

def anharmonic(t,x,v):
    k = 1
    alpha = 3
    F = -k*(x**alpha)
    return F

def damped(t,theta,omega):
    q = 1/2
    F = simpleHarmonic(t,theta,omega) - q*omega
    return F

def dampedDriven(t,theta,omega):
    fd = 1.2
    od = 2/3
    F = damped(t,theta,omega) + fd*np.sin(od*t)
    return F

def chaoticNonlinear(t,theta,omega):
    F = dampedDriven(t,np.sin(theta),omega)
    return F

def duffing(t,x,v):
    """
    https://en.wikipedia.org/wiki/Duffing_equation
    """
    delta = 0.3
    alpha = -1
    beta = 1
    gamma = 0.5
    omega = 1.2
    F = gamma*np.cos(omega*t) - delta*v - alpha*x - beta*x**3
    return F

def rayleighLorentz(t,x,v):
    """
    https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Lorentz_pendulum
    """
    omega = t/100
    F = -(omega**2)*x
    return F

# Simulation
def swing(dt):
    # initialize time
    t = np.arange(0,60,dt)

    initVelocity = 0
    initPosition = 0.2
    
    # function to observe
    pendulum = rayleighLorentz

    # initialize array
    positions = np.zeros(len(t))
    velocities = np.zeros(len(t))

    # initial conditions
    velocities[0] = initVelocity
    positions[0] = initPosition


    # Euler method (first order differential)
    # Euler-Cromer method (second order differential)
    # Approximates next value using Euler/Euler-Cromer method
    # y[n+1] = y[n] + h*f(t[n],y[n])
    for i in range(len(t)-1):
        velocities[i+1] = velocities[i] + dt*pendulum(t[i],positions[i],velocities[i])
        positions[i+1] = positions[i] + dt*velocities[i+1]
    # plt.plot(t,positions)

    # Reinitialize
    positions = np.zeros(len(t))
    velocities = np.zeros(len(t))
    velocities[0] = initVelocity
    positions[0] = initPosition

    # Midpoint method
    # y[n+1] = y[n] + h*f(t[n]+h/2,y[n]+(h/2)*f(t[n],y[n]))
    for i in range(len(t)-1):
        velocities[i+1] = velocities[i] + dt*pendulum(t[i]+dt/2,
                                                            positions[i]+(dt/2)*pendulum(t[i],positions[i],velocities[i]),
                                                            velocities[i]+(dt/2)*pendulum(t[i],positions[i],velocities[i]))
        positions[i+1] = positions[i] + dt*velocities[i+1]
    # plt.plot(t,positions)

    # Reinitialize
    positions = np.zeros(len(t))
    velocities = np.zeros(len(t))
    velocities[0] = initVelocity
    positions[0] = initPosition

    # Runge-Kutta method (RK4)
    # y[n+1] = y[n] + 1/6(k1+2k2+2k3+k4)
    # k1 = h*f(t[n],y[n])
    # k2 = h*f(t[n]+h/2,y[n]+h/2)
    # k3 = h*f(t[n]+h/2,y[n]+k2/2)
    # k4 = h*f(t[n]+h,y[n]+k3)
    for i in range(len(t)-1):
        k1 = dt*pendulum(t[i],positions[i],velocities[i])
        k2 = dt*pendulum(t[i]+dt/2,positions[i]+k1/2,velocities[i]+k1/2)
        k3 = dt*pendulum(t[i]+dt/2,positions[i]+k2/2,velocities[i]+k2/2)
        k4 = dt*pendulum(t[i]+dt,positions[i]+k3,velocities[i]+k3)
        velocities[i+1] = velocities[i] + (1/6)*(k1+2*k2+2*k3+k4)
        positions[i+1] = positions[i] + dt*velocities[i+1]
    plt.plot(t,positions)

    plt.show()

    return None

def main():
    # Lower time step value gives a more accurate solution
    # but also increases memory usage and running time
    TIME_STEP = 0.1
    swing(TIME_STEP)
    return None

if __name__ == "__main__":
    main()