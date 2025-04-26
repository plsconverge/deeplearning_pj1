from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step(self):
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma
        self.stage = 0

    def step(self) -> None:
        self.step_count += 1
        if self.stage >= len(self.milestones):
            return
        if self.step_count >= self.milestones[self.stage]:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0
            self.stage = self.stage + 1

class ExponentialLR(scheduler):
    def __init__(self, optimizer, init_step, lambda_pow=1):
        super().__init__(optimizer)
        self.init_step = init_step
        self.multiplier = np.exp(-lambda_pow)

    def step(self):
        self.step_count += 1
        self.optimizer.init_lr *= self.multiplier
