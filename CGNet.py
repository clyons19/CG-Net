# CG-Net.py 
# * Created by: Carter Lyons

# Contains:
# * Deep neural network implementation of Compound Gaussian Network (CG-Net) form by applying algorithm unrolling to the Compound Gaussian Least Squares (CG-LS) iterative algorithm. Estimates signal representation coefficients from linear measurements.
# * For more information reference:
#     1. Lyons C., Raj R. G., and Cheney M. (2023). "A Compound Gaussian Iterative Algorithm and Unrolled Network for Linear Inverse Problems," In-Review.
#     2. Lyons C., Raj R. G., and Cheney M. (2022). "CG-Net: A Compound Gaussian Prior Based Unrolled Imaging Network," in *2022 IEEE Asia-Pacific Signal and Information Processing Association Annual Summit and Conference*, pp. 623-629.

# Data requirements:
# * To use this DNN each image $I$ should be scaled down by $I_{max}$ i.e. the max pixel value of $I$. Then the wavelet coefficients $c$ and measurements $y$ should be created off of the scaled images

# Package Requirements (in addition to those required in Initialize.py):
# * Tensorflow 2.5.0

import tensorflow as tf
import numpy as np

import DescentDirections


class Z_InitialLayer(tf.keras.layers.Layer): # Calculates z0 the initial z variable estimate 
    def __init__(self, A, normalize = False, name = None): # Inputs: Measurement matrix A (tensor with dtype 'float32'). Whether to normalize the measurement matrix (boolean). Name of network (string), note name is required for saving network variables. 
        super(Z_InitialLayer, self).__init__(name = name)
        if normalize:
            self.A = A/tf.linalg.norm(A, ord = 2)
        else:
            self.A = A
        
    def call(self, inputs):
        return inputs@self.A   # Backprojection (A^T)y

class U_Layer(tf.keras.layers.Layer):  # Calculate the Tikhonov solution estimate of u
    def __init__(self, m, lamb, eps = 1e-3, name = None):  # Inputs: Measurement dimension (int). Regularization parameter lamb (real value). Stabilizing paramter eps (real value). Name of network (string), note name is required for saving network variables. 
        super(U_Layer, self).__init__(name = name)
        self.m = m  # Measurement dimension
        self.lamb = self.add_weight(name = '{}_lamb'.format(self.name), shape = (), initializer =  tf.keras.initializers.Constant(lamb), trainable = True)  # Regularization parameter lambda (learned by the network)  
        self.eps = self.add_weight(name = '{}_eps'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(eps), trainable = False)  # Stabilizing parameter epsilon (not learned by the network)

    def call(self, A, z, y):  # Inputs: Measurement matrix A (tensor of dtype 'float32'). Current z estimate. Measurements y.
        Azt = tf.linalg.diag(z)@tf.transpose(A)  # Calculate matrix (A*Diag(z))^T = (A_z)^T
        H = tf.linalg.cholesky(Azt@tf.transpose(Azt, perm = [0, 2, 1])+tf.math.maximum(self.lamb, self.eps)*tf.eye(self.m, dtype = float))  # Cholesky decomposition of matrix (A_z)^T(A_z)+lamb*I
        u = tf.linalg.cholesky_solve(H, Azt@tf.expand_dims(y, -1))  # Use Cholesky decomposition to find Tikhonov solution u = ((A_z)^T(A_z)+lamb*I)^(-1)((A_z)^Ty)
        return tf.reshape(u, tf.shape(z))
    
class U_LayerFast(tf.keras.layers.Layer):  # Calculate the Tikhonov solution estimate of u
    def __init__(self, m, lamb, eps = 1e-3, name = None):  # Inputs: Measurement dimension (int). Regularization parameter lamb (real value). Stabilizing paramter eps (real value). Name of network (string), note name is required for saving network variables.
        super(U_LayerFast, self).__init__(name = name)
        self.m = m  # Measurement dimension
        self.lamb = self.add_weight(name = '{}_lamb'.format(self.name), shape = (), initializer =  tf.keras.initializers.Constant(lamb), trainable = True)  # Regularization parameter lambda (learned by the network)  
        self.eps = self.add_weight(name = '{}_eps'.format(self.name), shape = (), initializer =  tf.keras.initializers.Constant(eps), trainable = False)  # Stabilizing parameter epsilon (not learned by the network)
    
    def call(self, A, z, y):  # Inputs: Measurement matrix A (tensor of dtype 'float32'). Current z estimate. Measurements y.
        Azt = tf.linalg.diag(z)@tf.transpose(A)  # Calculate matrix (A*Diag(z))^T = (A_z)^T
        H = tf.linalg.cholesky(tf.transpose(Azt, perm = [0, 2, 1])@Azt+tf.math.maximum(self.lamb, self.eps)*tf.eye(self.m, dtype = float)) # Cholesky decomposition on (A_z)(A_z)^T+lamb*I
        u = tf.linalg.cholesky_solve(H, tf.expand_dims(y, -1))  # Calculate ((A_z)(A_z)^T+lamb*I)^(-1)y 
        return tf.reshape(Azt@u, tf.shape(z))  # Find Tikhonov solution u = (A_z)^T((A_z)(A_z)^T+lamb*I)^(-1)y 
    
class Z_QuadraticDiagLayer(tf.keras.layers.Layer):
    def __init__(self, A, kappa, eta, q, eps = 1e-3, method = 'g', name = None):
        super(Z_QuadraticDiagLayer, self).__init__(name = name)
        self.size = tf.shape(A)[1]
        self.kappa = self.add_weight(name = '{}_kappa'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(kappa), trainable = True)
        self.eps = self.add_weight(name = '{}_eps'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(eps), trainable = False)
        self.eta = self.add_weight(name = '{}_eta'.format(self.name), shape = (self.size,1), initializer = tf.keras.initializers.Constant(eta), trainable = True)
        self.Q = self.add_weight(name = '{}_Q'.format(self.name), shape = (self.size,), initializer = tf.keras.initializers.Constant(q), trainable = True)
        if method == 'n':
            self.DescentMethod = DescentDirections.NewtonDescent(A, self.kappa, self.eps)
        elif method == 'g':
            self.DescentMethod = DescentDirections.GradientDescent(A, self.kappa)
        
    def call(self, z, u, y):
        return z + tf.reshape(self.eta*tf.linalg.diag(tf.math.maximum(self.Q, self.eps))@self.DescentMethod.UpdateDirection(z, u, y), tf.shape(z))

class Z_QuadraticTriuLayer(tf.keras.layers.Layer):  # Steepest descent update of z using PD quadratic matrix
    def __init__(self, A, kappa, eta, q, eps = 1e-3, method = 'g', name = None):  # Inputs: Measurement matrix A (tensor or matrix). Regularization paramter kappa (real value). Step size eta (real value). Stabilizing paramter eps (real value). Vector (size n(n+1)/2) q containing lower triangular entries of quadratic matrix. Name of network (string), note name is required for saving network variables. 
        super(Z_QuadraticTriuLayer, self).__init__(name = name)
        self.size = tf.shape(A)[1]
        self.kappa = self.add_weight(name = '{}_kappa'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(kappa), trainable = True)
        self.eps = self.add_weight(name = '{}_eps'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(eps), trainable = False)
        self.eta = self.add_weight(name = '{}_eta'.format(self.name), shape = (self.size,1), initializer = tf.keras.initializers.Constant(eta), trainable = True)
        self.q = self.add_weight(name = '{}_Q'.format(self.name), shape = (self.size*(self.size+1)//2,), initializer = tf.keras.initializers.Constant(q), trainable = True)
        if method == 'n':
            self.DescentMethod = DescentDirections.NewtonDescent(A, self.kappa, self.eps)
        elif method == 'g':
            self.DescentMethod = DescentDirections.GradientDescent(A, self.kappa)
        
    def call(self, z, u, y):
        U = tf.linalg.band_part(tf.reshape(tf.concat([self.q, self.q[self.size:][::-1]], 0), (self.size,self.size)), 0, -1)
        L, Q = tf.linalg.eigh((U+tf.transpose(U))/2) 
        return z + tf.reshape(self.eta*Q@tf.linalg.diag(tf.math.maximum(L, self.eps))@tf.transpose(Q)@self.DescentMethod.UpdateDirection(z, u, y), tf.shape(z))
  # Steepest descent update: z - eta*U*(Grad_z F)

class Z_QuadraticTriDiagLayer(tf.keras.layers.Layer):
    def __init__(self, A, kappa, eta, q, eps = 1e-3, method = 'g', name = None):
        super(Z_QuadraticTriDiagLayer, self).__init__(name = name)
        self.size = tf.shape(A)[1]
        self.kappa = self.add_weight(name = '{}_kappa'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(kappa), trainable = True)
        self.eps = self.add_weight(name = '{}_eps'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(eps), trainable = False)
        self.eta = self.add_weight(name = '{}_eta'.format(self.name), shape = (self.size,1), initializer = tf.keras.initializers.Constant(eta), trainable = True)
        self.diag = self.add_weight(name = '{}_diag'.format(self.name), shape = (self.size,), initializer = tf.keras.initializers.Constant(q[0]), trainable = True)
        self.offdiag = self.add_weight(name = '{}_offdiag'.format(self.name), shape = (self.size-1,), initializer = tf.keras.initializers.Constant(q[1]), trainable = True)
        if method == 'n':
            self.DescentMethod = DescentDirections.NewtonDescent(A, self.kappa, self.eps)
        elif method == 'g':
            self.DescentMethod = DescentDirections.GradientDescent(A, self.kappa)
        
    def call(self, z, u, y):
        L, Q = tf.linalg.eigh(tf.linalg.diag(self.diag)+tf.linalg.diag(self.offdiag,k = -1)+tf.linalg.diag(self.offdiag,k = 1)) 
        return z + tf.reshape(self.eta*Q@tf.linalg.diag(tf.math.maximum(L, self.eps))@tf.transpose(Q)@self.DescentMethod.UpdateDirection(z, u, y), tf.shape(z))
    
class Z_ProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, MinVal, MaxVal, name = None):
        super(Z_ProjectionLayer, self).__init__(name = name)
        self.min = self.add_weight(name = '{}_min'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(MinVal), trainable = True)
        self.max = self.add_weight(name = '{}_max'.format(self.name), shape = (), initializer = tf.keras.initializers.Constant(MaxVal), trainable = True)
        
    def call(self, inputs):
        return self.min+tf.nn.relu(inputs-self.min)-tf.nn.relu(inputs-self.max)

class CGNet:
    def __init__(self, K, J, A, style, lamb, kappa = 2, eta = 0.5, eps = 1e-3, method = 'g', normalize = True):
        self.m, n = tf.shape(A)
        self.A = A
        self.K = K
        self.J = J
        self.Mult = tf.keras.layers.multiply
        self.Initial =  Z_InitialLayer(A, normalize = normalize, name = 'z0')
        self.Project0 = Z_ProjectionLayer(1, np.exp(2), name = 'mReLU0')
        self.U0 = U_LayerFast(self.m, lamb, eps = eps, name = 'u_init')
        self.z_layers = list()
        self.Projects = list()
        self.u_layers = list()
        
        style = 'tridiag' if style not in ['diag', 'tridiag', 'full'] else style
        
        if style == 'diag':
            q = np.ones(n)
            Z_Layer = Z_QuadraticDiagLayer            
        elif style == 'tridiag':
            q = (np.ones(n), np.zeros(n-1))
            Z_Layer = Z_QuadraticTriDiagLayer         
        elif style == 'full':
            q = np.zeros((n*(n+1)//2,))
            indx = np.array([i*(n+1)+1 for i in range(n-n//2)]+[i*(n+1) for i in range(n-n//2-n%2, 0, -1)])-1
            q[indx] = np.ones(n)
            Z_Layer = Z_QuadraticTriuLayer
            
        for k in range(self.K):
            self.z_layers.append(list())
            self.Projects.append(list())
            for j in range(self.J):
                self.z_layers[k].append(Z_Layer(A, kappa, eta, q, eps = eps, method = method, name = 'z{}_{}'.format(k, j)))
                self.Projects[k].append(Z_ProjectionLayer(0.8, np.exp(3), name = 'mReLU{}_{}'.format(k,j)))
            self.u_layers.append(U_LayerFast(self.m, lamb, eps = eps, name = 'u{}'.format(k)))

    def call(self):
        inputs = tf.keras.Input(shape = (self.m, ))
        z = self.Initial(inputs)
        z = self.Project0(z)
        u = self.U0(self.A, z, inputs)
        for k in range(self.K):
            for j in range(self.J):
                z = self.z_layers[k][j](z, u, inputs)
                z = self.Projects[k][j](z)
            u = self.u_layers[k](self.A, z, inputs)
        outputs = self.Mult((z, u))
        model = tf.keras.Model(inputs, outputs, name = 'CG-Net')
        return model
    
    
class Grad:
    def __init__(self):
        return
    def call(self, model, loss, inputs, targets):
        with tf.GradientTape() as tape:
            estimates = model(inputs, training=True)
            loss_value = loss.call(targets, estimates)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)