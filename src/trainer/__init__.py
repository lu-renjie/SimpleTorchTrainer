"""
Trainer类用来对Agent进行训练, 如果训练方式不同, 可以实现不同的Trainer。
如果是多任务训练, 测试会比较麻烦, 可以加一个Evaluator来测试
"""

from .trainer_base import TrainerBase


