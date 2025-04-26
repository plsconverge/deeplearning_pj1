from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.moments = []

        # preset the moments to 0
        for layer in self.model.layers:
            if layer.optimizable == True:
                layer_moment = {}
                for key in layer.params.keys():
                    layer_moment[key] = np.zeros_like(layer.params[key])
                self.moments.append(layer_moment)
    
    def step(self):
        moment_count = 0
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    grad = layer.grads[key]
                    if layer.weight_decay:
                        # layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                        grad += layer.weight_decay_lambda * layer.params[key]
                    self.moments[moment_count][key] = self.mu * self.moments[moment_count][key] - self.init_lr * grad
                    # self.moments[moment_count][key] = self.mu * self.moments[moment_count][key] - self.init_lr * layer.grads[key]
                    layer.params[key] += self.moments[moment_count][key]
                moment_count += 1

        # moment_count = 0
        # flag_moments = self.moments is None
        # if flag_moments:
        #     self.moments = []
        # for layer in self.model.layers:
        #     if layer.optimizable == True:
        #         for key in layer.params.keys():
        #             if flag_moments:
        #                 self.moments.append(dict())
        #             para_temp = layer.params[key].copy()
        #             if layer.weight_decay:
        #                 layer.params[key] *= (1 - self.mu * layer.weight_decay_lambda)
        #             # if flag_moments:
        #             #     layer.params[key] -= self.init_lr * layer.grads[key] - self.mu * para_temp
        #             # else:
        #             #     layer.params[key] -= self.init_lr * layer.grads[key] - self.mu * self.moments[moment_count][key]
        #             delta_params = -self.init_lr * layer.grads[key]
        #             if not flag_moments:
        #                 delta_params += self.mu * self.moments[moment_count][key]
        #             layer.params[key] += delta_params
        #             # self.moments[moment_count][key] = layer.params[key] - para_temp
        #             self.moments[moment_count][key] = delta_params
        #         moment_count += 1
