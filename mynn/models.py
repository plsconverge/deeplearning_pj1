from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        # for i in range(len(self.size_list) - 1):
        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, conv_num=0, size_list=None, lambda_list=None):
        self.conv_num = conv_num
        self.size_list = size_list

        if size_list is not None:
            self.layers = []
            for i in range(conv_num):
                # convolution layer
                layer = conv2D(size_list[i*2][0], size_list[i*2][1], size_list[i*2][2])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)
                # activate function -- ReLU
                self.layers.append(ReLU())
                # pooling layer -- max
                if i < conv_num - 1:
                    layer_f = Pool(size_list[i*2+1])
                    self.layers.append(layer_f)
            # linear layer
            layer = Linear(size_list[-2][1], size_list[-1])
            if lambda_list is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[-1]
            self.layers.append(layer)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers[:len(self.layers)-1]:
            outputs = layer(outputs)
        outputs = self.layers[-1](np.squeeze(outputs, axis=(2, 3)))
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        grads = self.layers[-1].backward(grads)
        grads = grads.reshape(*grads.shape, 1, 1)
        for layer in reversed(self.layers[:len(self.layers)-1]):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            param_list = pickle.load(f)
        self.conv_num = param_list[0]
        self.size_list = param_list[1]

        self.layers = []
        for i in range(self.conv_num):
            # convolution layer
            layer = conv2D(self.size_list[i*2][0], self.size_list[i*2][1], self.size_list[i*2][2])
            layer.W = param_list[i+2]['W']
            layer.b = param_list[i+2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i+2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            self.layers.append(layer)
            # ReLU layer
            self.layers.append(ReLU())
            # pooling layer
            if i < self.conv_num - 1:
                self.layers.append(Pool(self.size_list[i*2+1]))
        # linear layer
        layer = Linear(self.size_list[-2][1], self.size_list[-1])
        layer.W = param_list[-1]['W']
        layer.b = param_list[-1]['b']
        layer.params['W'] = layer.W
        layer.params['b'] = layer.b
        layer.weight_decay = param_list[-1]['weight_decay']
        layer.weight_decay_lambda = param_list[-1]['lambda']
        self.layers.append(layer)
        
    def save_model(self, save_path):
        param_list = [self.conv_num, self.size_list]
        for i in range(self.conv_num):
            layer = self.layers[i*3]
            param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        layer = self.layers[-1]
        param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay, 'lambda' : layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN_withDropout(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """

    def __init__(self, conv_num=0, linear_num=0, size_list=None, lambda_list=None):
        self.conv_num = conv_num
        self.linear_num = linear_num
        self.size_list = size_list

        if size_list is not None:
            self.layers = []
            for i in range(conv_num):
                # convolution layer
                layer = conv2D(size_list[i * 2][0], size_list[i * 2][1], size_list[i * 2][2])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)
                # activate function -- ReLU
                self.layers.append(ReLU())
                # pooling layer -- max
                if i < conv_num - 1:
                    layer_f = Pool(size_list[i * 2 + 1])
                    self.layers.append(layer_f)
            input_dim = size_list[(conv_num-1)*2][1]
            for i in range(linear_num):
                # linear layer
                layer = Linear(input_dim, size_list[conv_num*2+i-1])
                input_dim = size_list[conv_num*2+i-1]
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[conv_num+i]
                self.layers.append(layer)
                if i < linear_num - 1:
                    # dropout
                    layer_f = Dropout(p=0.1)
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers[:self.conv_num*3-1]:
            outputs = layer(outputs)
        outputs = np.squeeze(outputs, axis=(2, 3))
        for layer in self.layers[self.conv_num*3-1:]:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        count = 0
        for layer in reversed(self.layers):
            count += 1
            if count == self.linear_num*2:
                grads = grads.reshape(*grads.shape, 1, 1)
            grads = layer.backward(grads)
        return grads

    def change_dropout(self):
        for i in range(self.linear_num-1):
            layer = self.layers[self.conv_num*3+i*2]
            layer.train_mode = (layer.train_mode == False)

    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            param_list = pickle.load(f)
        self.conv_num = param_list[0]
        self.linear_num = param_list[1]
        self.size_list = param_list[2]

        self.layers = []
        for i in range(self.conv_num):
            # convolution layer
            layer = conv2D(self.size_list[i * 2][0], self.size_list[i * 2][1], self.size_list[i * 2][2])
            layer.W = param_list[i + 3]['W']
            layer.b = param_list[i + 3]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 3]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 3]['lambda']
            self.layers.append(layer)
            # ReLU layer
            self.layers.append(ReLU())
            # pooling layer
            if i < self.conv_num - 1:
                self.layers.append(Pool(self.size_list[i * 2 + 1]))
        input_dim = self.size_list[self.conv_num*2-2][1]
        for i in range(self.linear_num):
            # linear layer
            pos = self.conv_num*2-1+i
            layer = Linear(input_dim, self.size_list[pos])
            input_dim = self.size_list[pos]
            pos = self.conv_num + 3 + i
            layer.W = param_list[pos]['W']
            layer.b = param_list[pos]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[pos]['weight_decay']
            layer.weight_decay_lambda = param_list[pos]['lambda']
            self.layers.append(layer)
            if i < self.linear_num - 1:
                # dropout layer
                self.layers.append(Dropout(p=0.1))

    def save_model(self, save_path):
        param_list = [self.conv_num, self.linear_num, self.size_list]
        for i in range(self.conv_num):
            layer = self.layers[i * 3]
            param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay,
                               'lambda': layer.weight_decay_lambda})
        for i in range(self.linear_num):
            layer = self.layers[self.conv_num*3+i*2-1]
            param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay,
                               'lambda': layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
