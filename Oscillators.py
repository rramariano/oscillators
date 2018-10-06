"""
Simulation of different kinds of oscillations
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Oscillator:

    #constant
    g = 9.8

    def __init__(self,dt=0.01,x=0.1,v=0.1):
        self.time = np.arange(0,60,dt)
        self.initX = x
        self.initV = v
        self.initConditions = (self.initX,self.initV)

    #methods
    #different kinds of oscillatory motion
    def simpleHarmonic(self,y,t,omega0=2):
        """
        d2x/dt2 + omega^2 * x = 0
        """
        x, v = y
        dydt = [v, -omega0**2 * x]
        return dydt

    def anharmonic(self,y,t,k=1.1,alpha=1.1):
        """
        F = -k*(x**alpha)
        """
        x,v = y
        dydt = [v,-k*(x**alpha)]
        return dydt

    def damped(self,y,t,omega0=2,beta=1.5):
        """
        d2x/dt2 + 2(beta)dx/dt + omega^2 * x = 0
        """
        x, v = y
        dydt = [v, self.simpleHarmonic(y,t)[1] - 2*beta*v]
        return dydt

    def dampedDriven(self,y,t,omega0=2,beta=1.5,A=1.1,omega=1.7):
        """
        d2x/dt2 + 2(beta)dx/dt + omega^2 * x = Acos(omega*t)
        """
        x,v = y
        dydt = [v, self.damped(y,t,omega0,beta)[1]+A*np.cos(omega*t)]
        return dydt

    def duffing(self,y,t,delta=0.02,alpha=1,beta=5,gamma=8,omega=0.5):
        """
        https://en.wikipedia.org/wiki/Duffing_equation
        """
        x, v = y
        dydt = [v, gamma*np.cos(omega*t) - delta*v - alpha*x - beta*x**3]
        return dydt

    def rayleighLorentz(self,y,t):
        """
        https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Lorentz_Oscillator
        """
        omega = np.exp(-0.1*t)*np.cos(t)
        x, v = y
        dydt = [v, (omega**2)*x]
        return dydt

    def vanDerPol(self,y,t,mu=1.5):
        """
        https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
        """
        x,v = y
        dydt = [v, mu*(1-x**2)*v-x]
        return dydt

    def move(self,function):
        return odeint(function, self.initConditions, self.time)

    def plotPosition(self,function):
        pos = self.move(function)[:,0]
        plt.plot(self.time,pos)
        plt.show()
        return None

    def plotVelocity(self,function):
        velocity = self.move(function)[:,1]
        plt.plot(self.time,velocity)
        plt.show()
        return None
    
    def phase(self,function):
        pos = self.move(function)[:,0]
        velocity = self.move(function)[:,1]
        plt.plot(pos,velocity)
        plt.show()
        return None

