"""This file contains a class for warmup and exponential decay learning rate lambda."""

from typing import Any


class WarmupExpDecayLRLambda:
    """This class warmup learning rate linearly and decay exponentially.

    Note: This class will be used as a lambda function for torch.optim.lr_scheduler.LambdaLR.
    """

    def __init__(
        self,
        base_lr: float,
        init_lr: float,
        max_lr: float,
        final_lr: float,
        warmup_steps: int,
        max_steps: float,
    ) -> None:
        """Initialize WarmupExpDecayLRLambda class.

        Args:
            base_lr (float): Base learning rate. This is same to optimizer's initial learning rate.
            init_lr (float): Initial learning rate. Returned at the first step.
            max_lr (float): Maximum learning rate. Returned at the last step of warmup.
            final_lr (float): Final learning rate. Returned after decay ends.
            warmup_steps (int): Warmup steps.
            max_steps (float): Maximum steps in training.
        """
        self.base_lr = base_lr
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gamma = (final_lr / max_lr) ** (1 / (max_steps - warmup_steps))

    def __call__(self, epoch: int) -> Any:
        """Call WarmupExpDecayLRLambda class.

        Args:
            epoch (int): Current epoch or step.

        Returns:
            float: Learning rate coefficient.
        """

        if epoch < self.warmup_steps:
            lr = self.init_lr + (self.max_lr - self.init_lr) * (epoch / self.warmup_steps)
        else:
            lr = max(self.final_lr, self.max_lr * self.gamma ** (epoch - self.warmup_steps))

        return lr / self.base_lr
