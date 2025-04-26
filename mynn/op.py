from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim)) * 1e-2
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        W = self.params['W']
        b = self.params['b']
        rtn = np.matmul(X, W) + b
        return rtn

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        W = self.params['W']
        self.grads['W'] = np.matmul(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return np.matmul(grad, W.T)

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * 0.1
        self.b = initialize_method(size=out_channels)

        self.C = out_channels
        self.K = kernel_size

        self.params = {'W' : self.W, 'b' : self.b}
        self.grads = {'W' : None, 'b' : None}
        self.input = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k] --- unknown use of unsqueezed dimension
        no padding
        """
        W = self.params['W']
        b = self.params['b']
        batch, in_channels, height, width = X.shape
        self.input = X

        X_col = np.lib.stride_tricks.as_strided(
            X,
            shape=(batch, in_channels, height-self.K+1, width-self.K+1, self.K, self.K),
            strides=(X.strides[0], X.strides[1], X.strides[2], X.strides[3], X.strides[2], X.strides[3])
        )
        X_col = X_col.transpose((0, 2, 3, 1, 4, 5)).reshape(batch, height-self.K+1, width-self.K+1, -1)
        W_col = W.reshape(self.C, -1)

        rtn = np.tensordot(X_col, W_col, axes=([3], [1])).transpose(0, 3, 1, 2)
        rtn += b.reshape(1, -1, 1, 1)
        return rtn

        # rtn = np.zeros((batch, self.C, height-self.K+1, width-self.K+1))
        # for _ in range(batch):
        #     for d in range(self.C):
        #         for i in range(height-self.K+1):
        #             for j in range(width-self.K+1):
        #                 rtn[_, d, i, j] += np.sum(X[_, :, i:i+self.K, j:j+self.K] * W[d, :, :, :])
        #         rtn[_, d, :, :] += b[d]
        # return rtn

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        output: [batch_size, channels, H, W]
        """
        W = self.params['W']
        X = self.input
        batch, _, hk, wk = grads.shape
        _, in_channels, height, width = X.shape
        # rtn = np.zeros(self.input.shape)
        # self.grads['W'] = np.zeros(W.shape)

        self.grads['b'] = np.sum(grads, axis=(0, 2, 3))

        X_col = np.lib.stride_tricks.as_strided(
            X,
            shape=(batch, in_channels, hk, wk, self.K, self.K),
            strides=(X.strides[0], X.strides[1], X.strides[2], X.strides[3], X.strides[2], X.strides[3])
        )
        X_col = X_col.transpose((0, 2, 3, 1, 4, 5)).reshape(batch, hk, wk, -1)
        grads_col = grads.transpose(1, 0 ,2, 3).reshape(self.C, -1)

        self.grads['W'] = np.dot(grads_col, X_col.reshape(-1, in_channels*self.K*self.K))
        self.grads['W'] = self.grads['W'].reshape(self.C, in_channels, self.K, self.K)

        grads_pad = np.zeros((batch, self.C, hk+(self.K-1)*2, wk+(self.K-1)*2))
        grads_pad[:, :, self.K-1:self.K-1+hk, self.K-1:self.K-1+wk] = grads

        W_rot = np.rot90(W, 2, axes=(2, 3))
        # W_rot = W_rot.transpose((1, 0, 2, 3))

        rtn = np.zeros_like(X)
        for i in range(self.K):
            for j in range(self.K):
                rtn[:, :, i:i+hk, j:j+wk] += np.tensordot(
                    grads_pad[:, :, i:i+hk, j:j+wk],
                    W_rot[:, :, i, j],
                    axes=([1], [0])
                ).transpose(0, 3, 1, 2)
        return rtn

        # W_rot = np.rot90(W, 2, axes=(2, 3))
        # for _ in range(batch):
        #     for d in range(self.C):
        #         for c in range(self.input.shape[1]):
        #             self.grads['W'][d, c] += np.correlate(grads[_, d, :, :], self.input[_, c, :, :], mode='valid')
        #             grads_pad = np.pad(grads[_, d, :, :], ((self.K-1, self.K-1), (self.K-1, self.K-1)))
        #             rtn[_, c] += np.convolve(grads_pad, W_rot[_, c, :, :], mode='valid')

        # for _ in range(batch):
        #     for c in range(rtn.shape[1]):
        #         for i in range(self.K):
        #             for j in range(self.K):
        #                 # gradient of weight matrix
        #                 self.grads['W'][:, c, i, j] += np.dot(
        #                     grads[_, :, :, :].reshape(self.C, -1),
        #                     self.input[_, c, i:i+hk, j:j+wk].reshape(-1)
        #                 )
        # for _ in range(batch):
        #     for c in range(self.C):
        #         for i in range(rtn.shape[2]):
        #             for j in range(rtn.shape[3]):
        #                 # window size
        #                 ind11 = np.min((self.K, i+1))
        #                 ind21 = np.min((self.K, j+1))
        #                 ind12 = np.min((i+1, hk))
        #                 ind22 = np.min((j+1, wk))
        #                 rtn[_, :, i, j] += np.dot(
        #                     W[c, :, i-ind12+1:ind11, j-ind22+1:ind21].reshape(rtn.shape[1], -1),
        #                     grads[_, c, i-ind11+1:ind12, j-ind21+1:ind22].reshape(-1)[::-1]
        #                 )
        # return rtn

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class Pool(Layer):
    def __init__(self, stride):
        self.stride = stride
        self.max_ind = None

        self.input_size = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        X: [batch_size, channels, H, W]
        """
        self.input_size = X.shape
        batch, in_channels, height, width = X.shape
        h_out, w_out = height // self.stride, width // self.stride
        X_col = np.lib.stride_tricks.as_strided(
            X,
            shape=(batch, in_channels, h_out, w_out, self.stride, self.stride),
            strides=(X.strides[0], X.strides[1], X.strides[2]*self.stride, X.strides[3]*self.stride, X.strides[2], X.strides[3])
        )
        X_col = X_col.reshape(batch, in_channels, h_out, w_out, -1)

        X_pool = np.max(X_col, axis=-1)
        X_ind = np.argmax(X_col, axis=-1)

        h_ind = (X_ind // self.stride) + np.arange(h_out)[:, np.newaxis] * self.stride
        w_ind = (X_ind % self.stride) + np.arange(w_out)[np.newaxis, :] * self.stride
        self.max_ind = (h_ind, w_ind)

        return X_pool

        # rtn = np.zeros((X.shape[0], X.shape[1], X.shape[2] // self.stride, X.shape[3] // self.stride))
        # self.max_ind = np.zeros_like(rtn)
        # for _ in range(X.shape[0]):
        #     for c in range(X.shape[1]):
        #         for i in range(rtn.shape[2]):
        #             for j in range(rtn.shape[3]):
        #                 rtn[_, c, i, j] = np.max(X[_, c, i*self.stride:(i+1)*self.stride, j*self.stride:(j+1)*self.stride])
        #                 self.max_ind[_, c, i, j] = np.argmax(X[_, c, i*self.stride:(i+1)*self.stride, j*self.stride:(j+1)*self.stride])
        # return rtn

    def backward(self, grads):
        rtn = np.zeros(self.input_size)
        batch, channels, _, _ = grads.shape

        h_ind, w_ind = self.max_ind
        for _ in range(batch):
            for c in range(channels):
                rtn[_, c, h_ind[_, c], w_ind[_, c]] = grads[_, c]
        return rtn

        # rtn = np.zeros(self.input.shape)
        # for _ in range(rtn.shape[0]):
        #     for c in range(rtn.shape[1]):
        #             for i in range(self.max_ind.shape[2]):
        #                 for j in range(self.max_ind.shape[3]):
        #                     ind1, ind2 = np.unravel_index(int(self.max_ind[_, c, i, j]), (self.stride, self.stride))
        #                     rtn[_, c, i*self.stride + ind1, j*self.stride + ind2] = grads[_, c, i, j]
        # return rtn
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.input = None
        self.grads = None
        self.labels = [str(_) for _ in range(max_classes)]

        self.has_softmax = True
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        if self.has_softmax:
            predicts = softmax(predicts)
        self.input = (predicts, labels)

        batch = predicts.shape[0]
        loss = -np.sum(np.log(
            predicts[np.arange(batch), labels.astype(int)]
        ))
        return loss / batch

        # loss = 0
        # for _ in range(predicts.shape[0]):
        #     loss -= np.log(predicts[_, int(labels[_])])
        # return loss / predicts.shape[0]

    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        predicts, labels = self.input
        batch = predicts.shape[0]

        self.grads = predicts.copy()
        self.grads[np.arange(batch), labels.astype(int)] -= 1
        self.grads /= batch

        # self.grads = self.input[0].copy()
        # for _ in range(self.input[0].shape[0]):
        #     max_ind = int(self.input[1][_])
        #     self.grads[_, max_ind] -= 1

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass

class Dropout(Layer):
    """
    Dropout layer
    """
    def __init__(self, p=0.1):
        self.p = p
        self.scale = 1 / (1 - p)
        self.zeros = None
        self.train_mode = True

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.train_mode:
            self.zeros = (np.random.random_sample(X.shape) > self.p)
            # self.zeros = (np.random.rand(*X.shape) > self.p)
            return X * self.zeros * self.scale
        else:
            return X

    def backward(self, grads):
        return grads * self.zeros * self.scale
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition