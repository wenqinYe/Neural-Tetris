import numpy as np
import matplotlib.pyplot as plt

import layers
reload(layers)

class Optimizer():
    def __init__(self, neural_net):
        """
        Optimizer constructor
        
        :param neural_net: Neural network to optimize
        """
        self.net = neural_net
        self.X = None
        self.Y = None
        
        self.error_fig = None
        self.error_ax = None
        
    def cross_entropy_loss(self, out, y):
        """
        Cross entropy
        
        :param out: the output of the neural network
        :param y: the expected output of the neural network
        
        :return: Cross entropy loss
        """
        return (np.multiply(y, np.log(out)) + np.multiply((1-y), np.log(out)))/self.net.batch_size
        
    def quadratic_loss(self, out, y):
        """
        Returns the quadratic loss derivative
        
        :param out: Output of the neural network
        :param y: Expected output
        
        :return: Quadratic loss derivative
        """
        return y - out
        
        
    def run_minibatch(self, X, Y, batch_size=1):
        """
        Runs minibatch on datset given.
        
        :param X: X inputs
        :param Y: Y inputs
        :param batch_size: Batch size (default is one)
        
        """
        self.net.batch_size = batch_size
        self.X = X
        self.Y = Y
        
        for batch in self.generate_minibatches(X, Y, batch_size):
            x_batch = batch[0]
            y_batch = batch[1]
            
            for i in range(len(x_batch)):
                x = np.matrix(x_batch[i])
                y = np.matrix(y_batch[i])
                out = self.net.forward(x)
                delta = self.quadratic_loss(out, y)
                self.net.backwards(delta)
            self.net.apply_gradient_cache()
                    
        
        
    def generate_minibatches(self, X, Y, batch_size):
        """
        Generates minibatches iteratively
        
        :param X: Numpy array representing inptus
        :param Y: Numpy array representing outputs
        """
        X = np.array(X)
        Y = np.array(Y)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices)-batch_size+1, batch_size):
            batch_indices = indices[i: i+batch_size]
            yield X[batch_indices], Y[batch_indices]
            
    def generate_spiral_data(self):
        """
        Generates a spiral dataset 
            Credit: http://cs231n.github.io/neural-networks-case-study/
        
        :return X, Y: Tuple of spiral data set X and Y coordiantes.
        """
        N = 100 #number of points per class
        D = 2 #dimensions
        K = 3 #number of classes
        
        X = np.zeros((N*K, D))
        Y = np.zeros(N*K, dtype='uint8')
        for j in range(K):
            ix = range(N*j, N*(j+1))
            
            r = np.linspace(0.0, 1, N) #radius
            t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) *0.2
            
            X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            Y[ix] = j
        return X, Y
        
    def training_error(self):
        """
        Computes training error
        
        :return: Training error
        """
        delta = 0
        total_d = 0
        for i in range(len(self.X)):
            out = self.net.forward(self.X[i])
    
            actual = self.Y[i]
            total_d += np.sum(np.abs(out-actual))
        
        return total_d / self.net.batch_size
        
    def start_error_plot(self):
        """
        Creates an error plot to visualize training error
        """
        self.error_fig = plt.figure()
        self.error_ax = self.error_fig.add_subplot('111')
        self.error_ax.plot([], [], 'r-')
        
        plt.show()
        
    def update_error_plot(self):
        """
        Update error plot based on current training error
        """
        line = self.error_ax.lines[0]
        error = self.training_error()
        
        last_x = None
        x_data = line.get_xdata()
        if(len(x_data) == 0):
            last_x = -1
        else:
            last_x = x_data[len(x_data)-1]
        line.set_xdata(np.append(x_data, last_x + 1))
 
 
        y_data = line.get_ydata()
        line.set_ydata(np.append(y_data, error))
        
        self.error_ax.relim()
        self.error_ax.autoscale_view()
        
        self.error_fig.canvas.draw()
        self.error_fig.canvas.flush_events()
        
    

def optimizer_test():
    """
    Manually verify the functionality of the optimizer + neural network.
    """
    net = layers.NeuralNet(layers=[['input', 5], ['tanh', 20], ['linear', 3]])
    optimizer = Optimizer(net)
    X = []
    Y = []
    for i in range(10):
        X.append(np.matrix(np.zeros(5)))
        Y.append(np.matrix(np.ones(3)))
    for i in range(10):
        X.append(np.matrix(np.ones(5)))
        y = np.matrix(np.zeros(3))
        y[:, 1] = 1
        Y.append(y)
                
    optimizer.start_error_plot()
    net.alpha_weight = 1e-4
    net.alpha_bias = 1e-4
    net.adagrad_decay = 0.99
    i = 0
    for i in range(10000):
        optimizer.run_minibatch(X, Y, batch_size=10)
        i+=1 
    
        if i % 100 == 0:
            print ""
            optimizer.update_error_plot()
            print net.forward(X[13].ravel())
            print Y[13]
            print net.forward(X[0].ravel())
            print Y[0]
 
