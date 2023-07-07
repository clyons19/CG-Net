# DescentDirections.py
# * Created by: Carter Lyons
# 
# Contains:
# * Functions defining inverse nonlinearity and its derivatives
# * Descent Directions for steepest descent
# 
# Package Requirements (in addition to those required in Initialize.py):
# * Tensorflow 2.5.0

import tensorflow as tf
import numpy as np

class GradientDescent: # (i.e. steepest descent using Euclidean norm)
    def __init__(self, A, kappa):  # Inputs: Measurement matrix A (tensor or matrix). Regularization paramter kappa (real value).
        self.A = A
        self.kappa = kappa
        
    def f(self, z):  # Inverse nonlinearity between z and x in compound Gaussian prior (we use f(z) = log(z))
        return tf.math.log(z)
    
    def df(self, z):  # Derivative of inverse nonlinearity. Used in calculating the gradient.
        return 1/z
        
    def UpdateDirection(self, z, u, y):
        Aut = tf.linalg.diag(u)@tf.transpose(self.A) # Calculate (A_u)^T
        return 2*Aut@(tf.expand_dims(y, -1)-tf.transpose(Aut, perm = [0, 2, 1])@tf.expand_dims(z, -1))-2*tf.expand_dims(self.kappa*self.df(z)*self.f(z), -1)  # Negative gradient of F(u, z) w.r.t z

class NewtonDescent: # (i.e. steepest descent using quadratic norm defined by Hessian)
    def __init__(self, A, kappa, eps):
        self.A = A
        self.kappa = kappa
        self.eps = eps
        
    def f(self, z):  # Inverse nonlinearity between z and x in compound Gaussian prior (we use f(z) = log(z))
        return tf.math.log(z)
    
    def df(self, z):  # Derivative of inverse nonlinearity. Used in calculating the gradient.
        return 1/z
        
    def ddf(self, z):  # Second derivative of inverse nonlinearity. Used in calculating the Hessian.
        return -1/z**2
    
    def UpdateDirection(self, z, u, y):
        Aut = tf.linalg.diag(u)@tf.transpose(self.A)
        AtA = Aut@tf.transpose(Aut, perm = [0, 2, 1]) 
        zshape = tf.shape(z)
        yshape = tf.shape(y) 
        H = tf.linalg.cholesky(AtA + self.kappa*tf.linalg.diag(self.ddf(z)*self.f(z)+self.df(z)**2))      
#         L, Q = tf.linalg.eigh(H)
#         H = Q@tf.linalg.diag(tf.math.maximum(L, self.eps))@tf.transpose(Q, perm = [0, 2, 1])
        return tf.linalg.cholesky_solve(H, Aut@tf.expand_dims(y, -1) - AtA@tf.expand_dims(z, -1) - tf.expand_dims(self.kappa*self.df(z)*self.f(z), -1))