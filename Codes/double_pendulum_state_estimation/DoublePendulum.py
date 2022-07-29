import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
from PIL import Image
from PIL import ImageDraw
import ImageGen as noise_gen

class DoublePendulum:
    
    def __init__(self, episodes, max_length, img_size, seed):
        # Maximum time, time point spacings and the time grid (all in s).
        self.state_dim = 4
        self.episodes = episodes
        self.tmax, self.dt = max_length*0.01, 0.01
        self.t = np.arange(0, self.tmax, self.dt)
        self.img_size_internal = 128
        self.x0 = self.y0 = 64
        self.plt_length = 30 
        self.plt_width = 5
        self.img_size = img_size
        self.length = 1
        self.random = np.random.RandomState(seed)
        
        # Pendulum rod lengths (m), bob masses (kg).
        self.L1, self.L2 = 1, 1
        self.m1, self.m2 = 1, 1
        
        # The gravitational acceleration (m.s-2).
        self.g = 9.81
        
    def calc_E(self, y):
        """Return the total energy of the system."""

        th1, th1d, th2, th2d = y.T
        V = -(self.m1+self.m2)*self.L1*self.g*np.cos(th1) - self.m2*self.L2*self.g*np.cos(th2)
        T = 0.5*self.m1*(self.L1*th1d)**2 + 0.5*self.m2*((self.L1*th1d)**2 + (self.L2*th2d)**2 +
                2*self.L1*self.L2*th1d*th2d*np.cos(th1-th2))
        return T + V
    
    def deriv(self, y, t, L1, L2, m1, m2):
        """Return the first derivatives of y = theta1, z1, theta2, z2."""
        theta1, z1, theta2, z2 = y

        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

        theta1dot = z1
        z1dot = (self.m2*self.g*np.sin(theta2)*c - self.m2*s*(self.L1*z1**2*c + self.L2*z2**2) -
                 (self.m1+self.m2)*self.g*np.sin(theta1)) / self.L1 / (self.m1 + self.m2*s**2)
        theta2dot = z2
        z2dot = ((self.m1+self.m2)*(self.L1*z1**2*s - self.g*np.sin(theta2) + self.g*np.sin(theta1)*c) + 
                 self.m2*self.L2*z2**2*s*c) / self.L2 / (self.m1 + self.m2*s**2)
        return theta1dot, z1dot, theta2dot, z2dot
    
    def initial_state(self):
        # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
        return  np.array([2*random.uniform(0, 1)*np.pi/7, 0, 2*random.uniform(0, 1)*np.pi/4, 0])

    def datagen(self):
        targets = np.zeros(shape= [self.episodes, len(self.t)] + [self.state_dim])
        imgs = np.zeros(shape= [self.episodes, len(self.t)] + [self.img_size, self.img_size], dtype=np.uint8)
        for i in range(self.episodes):
            # Do the numerical integration of the equations of motion
            y0 = self.initial_state()
            y = odeint(self.deriv, y0, self.t, args=(self.L1, self.L2, self.m1, self.m2))

            # Check that the calculation conserves total energy to within some tolerance.
            EDRIFT = 0.2
            
            # Total energy from the initial conditions
            E = self.calc_E(y0)
            if np.max(np.sum(np.abs(self.calc_E(y) - E))) > EDRIFT:
                sys.exit('Maximum energy drift of {} exceeded.'.format(EDRIFT))

            # Unpack z and theta as a function of time
            theta1, theta2 = y[:,0], y[:,2]

            # Convert to Cartesian coordinates of the two bob positions.
            x1 = self.L1 * np.sin(theta1)
            y1 = -self.L1 * np.cos(theta1)
            x2 = x1 + self.L2 * np.sin(theta2)
            y2 = y1 - self.L2 * np.cos(theta2)
            targets[i] = np.concatenate((np.expand_dims(x1, axis=-1),np.expand_dims(y1, axis=-1),
                                        np.expand_dims(x2, axis=-1),np.expand_dims(y2, axis=-1)),axis = -1)
            for j in range(len(self.t)):
                imgs[i, j] = self._generate_single_image([x1[j], y1[j], x2[j], y2[j]])
        return imgs, targets
    
    def _generate_single_image(self, pos):
        x1 =- pos[0] * (self.plt_length / self.length) + self.x0
        y1 =- pos[1] * (self.plt_length / self.length) + self.y0
        x2 =- pos[2] * (self.plt_length / self.length) + self.x0
        y2 =- pos[3] * (self.plt_length / self.length) + self.y0
        img = Image.new('F', (self.img_size_internal, self.img_size_internal), 0.0)
        draw = ImageDraw.Draw(img)
        
        draw.line([(self.x0, self.y0), (x1, y1)], fill=1.0, width=self.plt_width)
        draw.line([(x1, y1), (x2, y2)], fill=1.0, width=self.plt_width)
        

        img = img.resize((self.img_size, self.img_size), resample=Image.ANTIALIAS)
        img_as_array = np.asarray(img)
        img_as_array = np.clip(img_as_array, 0, 1)
        return 255.0 * img_as_array
    
    def add_observation_noise(self, imgs, first_n_clean, corr=0.2, lowlow=0.1, lowup=0.4, uplow=0.6, upup=0.9):
        return noise_gen.add_img_noise(imgs, first_n_clean, self.random, corr, lowlow, lowup, uplow, upup)
                


