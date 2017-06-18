import numpy as np
import skimage
from skimage.util import view_as_windows as viewW

class NeuralNet():
    def __init__(self, layers=[]):
        """
        Neural Net constructor
        
        :param layers: - An input of layer specifications
                 Size input goes by (height, width)
                 Example: [['input', {"size": (1, 100)}], ['relu', {"size": (1,10)}], ['conv', {"size": (100, 100), 
                            "window_size":(2,2), "stride":1, "activation_func":'relu'}]]
        """
        self.layers = layers
        self.weights = []
        self.weights_cache = [] #used for gradient checking
        for i in range(1, len(self.layers)):
            if self.layers[i][0] == "conv":
                height, width = self.layers[i][1]["window_size"]
                conv_weight = np.random.normal(0, 1, size=(1, height * width)) * 1/np.sqrt(height * width)
                self.weights.append(conv_weight)
            else:
                number_inputs = self.layers[i][1]
                previous_size = self.layers[i-1][1]["size"][0] * self.layers[i-1][1]["size"][1] 
                this_size = self.layers[i][1]["size"][0] * self.layers[i][1]["size"][1]
                weight_matrix = np.random.normal(0, 1, size=(previous_size, this_size)) * 1/np.sqrt(previous_size)
                self.weights.append(weight_matrix)
                
        self.biases = []
        self.biases_cache = [] #used for gradient checking
        for i in range(1, len(self.layers)):
            height, width = self.layers[i][1]["size"]
            bias_vector = np.zeros((1, height * width))
            self.biases.append(bias_vector)
            
        self.activation_func = {
            'relu': self.relu,
            'tanh': self.tanh,
            'ln': self.ln,
            'exp': self.exp,
            'linear': self.linear,
            'batchnorm': self.batchnorm
        }
        self.activation_deriv = {
            'relu': self.relu_derivative,
            'tanh': self.tanh_derivative,
            'ln': self.ln_derivative,
            'exp': self.exp_derivative,
            'linear': self.linear_derivative,
            'batchnorm': self.batchnorm_derivative
        }
        
        self.outputs_cache = []
        
        self.weight_gradients_cache = []
        self.bias_gradients_cache = []
        self.reset_gradient_cache()
        
        self.weight_adagrad_cache = []
        self.adagrad_decay = 0.99
        self.init_adagrad_cache()
        
        self.convolution_cache = []
        
        self.alpha_weight = 0.1
        self.alpha_bias = 0.1
        
        self.batchnorm_gamma = 1
        self.batchnorm_beta = 0
        
        self.batch_size = None #needed when applying the accumulated gradients
        
        
    def reset_gradient_cache(self):
        """
        Resets the gradient cache.
        """
        self.weight_gradients_cache = []
        self.bias_gradients_cache = []
        
        for weight in self.weights:
            self.weight_gradients_cache.append(np.zeros_like(weight))
        for bias in self.biases:
            self.bias_gradients_cache.append(np.zeros_like(bias))
            
    def init_adagrad_cache(self):
        """
        Initializes the adagrad cache 
        """
        self.weight_adagrad_cache = []
        for weight in self.weights:
            self.weight_adagrad_cache.append(np.zeros_like(weight))
            
    def apply_gradient_cache(self):
        """
        Applys the accumulated gradients from backpropagation.
        """
        for i in range(len(self.weights)):
            weight_update = self.weight_gradients_cache[i]/self.batch_size
            weight_update = weight_update / np.sqrt(self.weight_adagrad_cache[i] + 1e-6)
            self.weights[i] += weight_update
            
            self.biases[i] += self.bias_gradients_cache[i] / self.batch_size
        
        self.reset_gradient_cache()
        
    def relu(self, vec):
        """
        Relu activation function.
        
        :param vec: Vector to apply relu activation to.
        :return: Vector with relu activation applied
        """        
        out = np.where(vec<0, 0, vec)
        return out
        
    def tanh(self, vec):
        """
        tanh activation function.
        
        :param vec: Vector to apply relu activation to.
        :return: Vector with tanh activation applied
        """        
        return np.tanh(vec)
        
    def ln(self, vec):
        """
        ln activation function.
        
        :param vec: Vector to apply ln activation to.
        :return: Vector with ln activation applied
        """
        return np.log(vec)
        
    def exp(self, vec):
        """
        exp activation function.
        
        :param vec: Vector to apply exp activation to.
        :return: Vector with exp activation applied
        """
        return np.multiply(0.001, np.exp(vec))
        
    def linear(self, vec):
        """
        Linear activation function
        
        :param vec: Vector to apply linear activation to.
        :return: Vector with linear activation applied
        """
        return vec
        
        
    def softmax(self, vec):
        """
        Softmax normalization
        
        :param vec: Vector to apply softmax to
        :return: Vector with softmax applied
        """
        vec -= np.max(vec)
        return np.exp(vec) / np.sum(np.exp(vec))
        
    def batchnorm(self, vec):
        """
        Applies batchnorm to layer (zero mean unit gaussian)
        
        :param vec: Vector of inputs to apply softmax
        :return: Output with batchnorm applied
        """
        mean = np.mean(vec)
        variance = np.sum(np.square(vec - mean)) / mean[:0].size
        print mean[:0].size
        
        norm = (vec - mean) / np.sqrt(variance + 1e-4)
        return self.batchnorm_gamma * norm + self.batchnorm_beta
        
    def convolution(self, matrix, weight_vector, options={"window_size":(1,1), "stride":1, "activation_func":'relu'}):
        """
        Convolves a 2d vector by a weight_vector.
        
        Transforms the 2d input, into row vectors that represent the convolutions
        """
        
        """
        Perserve the dimensions of the input
        """
        if options["stride"] > 1:
            raise "striding greater than 1 not implemented yet"
            
        padding_amount = (options["window_size"][0]-1) 
        padded_matrix = np.pad(matrix, (0, padding_amount), 'constant')
        conv_rows = self.im2col(padded_matrix, options["window_size"])
        conv_rows = np.multiply(conv_rows, weight_vector[0])
        
        """
        Sum each convolution as an input into a neuron
        """ 
        conv_rows = np.sum(np.matrix(conv_rows), axis=1) 
        return conv_rows.T
        
    def relu_derivative(self, output):
        """
        Applies the relu derivative
        
        :param output: The output cache
        :return: The derivative of the output
        """        
        deriv = np.where(output < 0, 0, output)
        deriv = np.where(deriv > 0, 1, deriv)
        return deriv
        
    def tanh_derivative(self, output):
        """
        Applies the tanh derivative
        
        :param output: The output cache
        :return: The derivative of the output
        """        
        return 1 - np.square(output)
        
    def ln_derivative(self, output):
        """
        Applies the ln derivative
        
        :param output: The output cache
        :return: The derivative of the output
        """        
        return 1/output
        
    def exp_derivative(self, output):
        """
        Applies the exp derivative
        
        :param output: The output cache
        :return: The derivative of the output
        """
        deriv = np.where(output != 0, 1, output)
        
    def linear_derivative(self, output):
        """
        Applies the linear derivative
        
        :param output: The output cache
        :return: The derivative of the output
        """
        return np.ones_like(output)
        
    def batchnorm_derivative(self, outout):
        """
        Applies the batchnorm derivative
            NOTE: TO BE IMPLEMENTED! Was planning on using 
                  batchnorm but i'm still learning how to code it.   
        
        :param output: The output cache
        :return: The derivative of the output
        """
        pass

        
    def equal_conv_pad(self, matrix, options):
        """
        Pads a matrix based on the stepsize, and focal size so that
        the output of the conv net will be exactly the size of the matrix
        """
        padding_amount = (options["window_size"][0]-1) 
        padded_matrix = np.pad(matrix, (0, padding_amount), 'constant')
        
        return padded_matrix
        
        
    def forward(self, X, use_parameter_copy=False):
        """
        X - Vector of inputs into the neural net
            Example [[3, 2, 1]]
        cache_output - Caches the output of each layer. Useed for back propagation.
        parameter copy - Copied parameters (weights and biases) are used for gradient checking.
        """
        self.outputs_cache = []
        
        """
        Sometimes the vector input will be a matrix [[3,2,1]]
        othertimes it will be just an array [3, 2, 1]
        So the program needs to convert 1d arrays into matrix arrays
        """
        self.outputs_cache.append(np.matrix(X))
            
        previous_layer_input = self.outputs_cache[0]
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if layer[0] == 'input':
                continue              
                
            weight_matrix = self.weights[i-1]
            bias = self.biases[i-1]
            
            if(use_parameter_copy):
                weight_matrix = self.weights_copy[i-1]
                bias = self.biases_copy[i-1]
                
            if layer[0] == 'conv':
                opts = layer[1]
                height, width = self.layers[i-1][1]["size"]
                
                previous_layer_input = previous_layer_input.reshape(height, width)
                summed_inputs = self.convolution(previous_layer_input, self.weights[i-1], opts) + bias
                summed_inputs = summed_inputs.reshape(height, width)
                activation_func = opts["activation_func"]
            else:
                """
                Weights are off by one, but not layers.
                """
                previous_height, previous_width = self.layers[i-1][1]["size"]
                previous_layer_input = previous_layer_input.reshape(1, previous_height * previous_width)
                summed_inputs = np.dot(previous_layer_input, weight_matrix) + bias
                activation_func = layer[0]
                
            previous_layer_input = self.activation_func[activation_func](summed_inputs)
            self.outputs_cache.append(previous_layer_input)
            
        return previous_layer_input
        
    def backwards(self, error):
        """
        Runs a backward pass of the neural network (using backprop + adagrad)
        
        :param error: The error in the output of the neural network
        """
        error_gradient = np.matrix(error)
        
        for i in range(len(self.layers)-1, 0, -1):

            layer = self.layers[i]
            activation_func = layer[0]
            output = self.outputs_cache[i]
            options = layer[1]
            size = layer[1]["size"]
            previous_output = np.matrix(self.outputs_cache[i-1].ravel())
            
            bias_gradient = error_gradient
            
            if layer[0] == 'conv':                
                previous_output_2d = self.equal_conv_pad(previous_output.reshape(size[0], size[1]), options)
                
                row_convs = self.im2col(previous_output_2d, layer[1]["window_size"], layer[1]["stride"])
                
                weight_gradient = np.dot(previous_output, row_convs)
                
                error_matrix = error_gradient.reshape(size[0], size[1])
                error_row_convs = self.im2col(error_matrix, layer[1]["window_size"], layer[1]["stride"])
                
                error_gradient = self.convolution(error_matrix, self.weights[i-1], options)               
            else:
                activation_derivative = self.activation_deriv[activation_func](output)
                
                """
                Get the real error gradient by multiplying it by the activation function derivative
                """
                error_gradient = np.multiply(error_gradient, activation_derivative)

                weight_gradient = np.dot(np.transpose(previous_output), error_gradient)
                """
                i-1 because the weights_gradient lacks the input layer so it's index
                will be one less.
                """
                
                error_gradient = np.dot(error_gradient, np.transpose(self.weights[i-1]))
            
            self.weight_gradients_cache[i-1] += weight_gradient * self.alpha_weight
            self.bias_gradients_cache[i-1] += bias_gradient * self.alpha_bias

            self.weight_adagrad_cache[i-1] = self.adagrad_decay * self.weight_adagrad_cache[i-1] + (1-self.adagrad_decay) * np.square(weight_gradient)
                    
                    
    def im2col(self, A, BSZ, stepsize=1): 
        """
        Allows for matrix multiplication of convolution. Turns matrix
        into column vectors representing the sliding windows.
            Credit: https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
        
        :param A: Matrix
        :param BSZ: Batch size
        :param stepsize: Step size
        
        :return: Row vectors representing convolutions.
        """ 
        return np.transpose(viewW(A, (BSZ[0], BSZ[1])).reshape(-1, BSZ[0] * BSZ[1]).T[:, ::stepsize])
        
    def load_parameters(self, base_path_name):
        """
        Loads saved parameters from a specifid base file path.
        
        :param base_path_name: Base path name to load paramers.
        """
        for i in range(len(self.layers)-1):
            self.weights[i] = np.load(base_path_name+"_weight_" + str(i)+".npy")
            self.biases[i] = np.load(base_path_name+"_bias_"+str(i)+".npy")
            
    def save_parameters(self, base_path_name):
        """
        Saves parameters from a specifid base file path.
        
        :param base_path_name: Base path name to save paramers to.
        """
        for i in range(len(self.layers)-1):
            np.save(base_path_name+"_weight_" + str(i)+".npy", self.weights[i])
            np.save(base_path_name+"_bias_"+str(i)+".npy", self.biases[i])
            
        