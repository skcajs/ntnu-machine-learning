import matplotlib
import sympy
import matplotlib.pyplot as plt
import random

class GradientDescent:
    def __init__(self, fx):
        self.fx = fx

    def differentiate(self):
        x = sympy.Symbol('x')
        return self.fx.diff(x)



