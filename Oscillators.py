"""
Simulation of different kinds of oscillations
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Pendulum:

    #constant
    g = 9.8

    def __init__(self,dt=0.04,l=1,x=0.1,v=0.1,theta=0.2,omega=0):
        self.time = np.arange(0,60,dt)
        self.length = l
        self.initX = x
        self.initV = v
        self.initTheta = theta
        self.initOmega = omega

    #methods
    #different kinds of pendulum/oscillatory motion
    def simpleHarmonic(self,y,t):
        """
        Small angle approximation
        F = -(g*theta)/l
        theta(t0)=0.2
        omega(t0)=0
        """
        theta, omega = y
        dydt = [omega, -Pendulum.g*theta/self.length]
        return dydt

    def anharmonic(self,y,t,k,alpha):
        """
        F = -k*(x**alpha)
        """
        x,v = y
        dydt = [v,-k*(x**alpha)]
        return dydt

    def damped(self,y,t,q=1/2):
        """
        F = -(g*theta)/ - q*omega
        """
        theta, omega = y
        dydt = [omega, self.simpleHarmonic(y,t)[1] - q*omega]
        return dydt

    def dampedDriven(self,y,t,fd=1.2,od=2/3):
        """
        F = -(g*theta)/l - q*omega + fd*sin(od*t)
        """
        theta,omega = y
        dydt = [omega, self.damped(y,t)[1] + fd*np.sin(od*t)]
        return dydt

    def chaoticNonLinear(self,y,t):
        """
        F = -(g*sin(theta))/l - q*omega + fd*sin(od*t)
        """
        theta, omega = y
        theta = np.sin(y[0])
        z=(theta,omega)
        dydt = [omega, self.dampedDriven(z,t)[1]]
        return dydt

    def duffing(self,y,t,delta=0.3,alpha=-1,beta=1,gamma=0.5,omega=1.2):
        """
        https://en.wikipedia.org/wiki/Duffing_equation
        """
        x, v = y
        dydt = [v, gamma*np.cos(omega*t) - delta*v - alpha*x - beta**3]
        return dydt

    def rayleighLorentz(self,y,t,omega=1.2):
        """
        https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Lorentz_pendulum
        """
        x, v = y
        dydt = [v, (omega**2)*x]
        return dydt