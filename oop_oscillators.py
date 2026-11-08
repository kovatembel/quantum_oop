#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:25:59 2026

@author: yosi
"""

import scipy
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import functools

from numpy import sqrt, log, exp, sin, cos, floor, pi

NUM_LEVELS_OSCILLATOR = 3 # number of truncated levels to keep in a harmonic oscillator

class System:
    """
    Global system, containing information on all degrees of freedom,
    methods for calculating time-evolution and expectation values.
    """
    
    def __init__(self):
        self.oscillators = {} # each object is a dictionary with keys 
        #('obj', 'hilbertsp_ind', representing the object itself and its index in the system's Hilbert space)
        self.dof_dims = [] # keeps track of single-particle dimension of each dof
        self.couplings = []
        self.auto_label_index = 1 # for creating auto labels
        self.dofs_added = 0
        self.hilbertsp_dim = 1 # for every new DOF multiply this by the DOF single-level dimension
    
    def add_oscillator(self, m, omega, label=''):
        if label == '':
            label = f'obj_{self.auto_label_index}'
            self.auto_label_index += 1
            
        osc = Oscillator(m, omega, label)
        osc_dim = osc.dim
        hilbertsp_ind = self.dofs_added
        
        self.oscillators[label] = {'obj': osc, 'hilbertsp_ind': hilbertsp_ind}
        self.dofs_added += 1
        self.hilbertsp_dim = self.hilbertsp_dim * osc_dim
        self.dof_dims.append(osc_dim)
        
    # Add coupling ~ k_c/2 * (x_1 - x_2)^2 between two oscillators
    def add_oscillator_coupling(self, label_1, label_2, k_c):
        coupling = Coupling(label_1, label_2, k_c)
        self.couplings.append(coupling)
    
    def init_ground_state():
        pass
    
    def prepare_operator(self, op_name, dof_label):
        """
        Create the matrix for the operator acting on degree of freedom dof_label
        
        op_name can be either of:
            'x', 'p', 'c', 'c_dag'
        """
        
        single_particle_operator = self.oscillators[dof_label]['obj'].operator_matrix(op_name)
        dof_ind = self.oscillators[dof_label]['hilbertsp_ind']
        
        # find the dimension of truncated Hilbert space right and left to our dof                
        right_dim = 1
        for i in range(dof_ind):
            right_dim = right_dim * self.dof_dims[i]
        left_dim = 1
        for i in range(dof_ind+1,len(self.dof_dims)):
            left_dim = left_dim * self.dof_dims[i]            
        I_right = np.eye(right_dim)
        I_left = np.eye(left_dim)
        
        operator = functools.reduce(np.kron, [I_left, single_particle_operator, I_right])
        
        return operator
    
    def prepare_operator_time_evolution(self, t):
        pass
    
    def calc_expectation_val(self, operator):
        pass


class DOF:
    """
    Abstract class for representing a degree of freedom in the system.    
    """
    def __init__(self, dim, label):
        self.label = label
        self.dim = dim
    
class Oscillator(DOF):
    """
    Class for representing one oscillator.
    """
    
    def __init__(self, m, omega, label):
        dim = NUM_LEVELS_OSCILLATOR
        super().__init__(dim, label)
        self.m = m
        self.omega = omega
        
    def operator_matrix(self, op_name):
        """
        Returns the single-particle operator matrix for the given op_name,
        with op_name being one of the following:
            ('x', 'p', 'c', 'c_dagger')
        """
        
        op_name_list = ('x', 'p', 'c', 'c_dagger')
        if op_name not in op_name_list:
            raise Exception(f'Called operator_matrix in Oscillator object with unknown operator name {op_name}')
        
        c = np.diag(sqrt(np.arange(1,self.dim)),1)
        c_dagger = np.diag(sqrt(np.arange(1,self.dim)),-1)
        
        if op_name == 'c':
            return c
        elif op_name == 'c_dagger':
            return c_dagger
        elif op_name == 'x':
            x = 1/sqrt(2*self.m*self.omega)*(c+c_dagger)
            return x
        elif op_name == 'p':
            p = -1j*sqrt(self.m*self.omega/2)*(c-c_dagger)
            return p


class Coupling:
    """
    Class for representing coupling between two oscillators.
    """
    
    def __init__(self, label_1, label_2, k):
        self.label_1 = label_1
        self.label_2 = label_2
        self.k = k
        

main_sys = System()
main_sys.add_oscillator(1, 2, '1')
main_sys.add_oscillator(1, 5, '2')
main_sys.add_oscillator(1, 3, '3')
main_sys.add_oscillator_coupling('1', '2', 4)


I1 = np.eye(2)
sigma_x = np.array([[0,1],[1,0]])

b = np.kron(I1, sigma_x)
c = np.kron(sigma_x, I1)