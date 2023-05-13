import scipy as sp
import numpy as np
import pysr

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import (wl, wlexpr, Global)

session = WolframLanguageSession("/home/dex/Mathematica/12.0/Executables/WolframKernel")

class Function:
    """
    This class is used to create a Wolfram Language string from a given right-hand side (rhs) expression. It also provides methods for setting the string and numerically integrating the function.

    __init__(self, rhs, prototype="f(x)")
        Initializes the Function class with the given rhs expression and prototype.

    wl_create_string(self, rhs)
        Creates a Wolfram Language string from the given rhs expression.

    wl_set_string(self, string)
        Sets the Wolfram Language string to the given string.

    wl_nintegrate(self)
        Numerically integrates the function using Wolfram.

    __str__(self)
    Returns a string representation of the function.
    """

    def wl_nintegrate(self, constraint_x=0.0, constraint_y=0.0, start_x=0.0, end_x=1.0, dx=0.01):
        session.evaluate(
            wlexpr(
                f'''
                    function[point_] :=
                      NDSolveValue[{{ y'[x] == {self.wlrhs}, y[{constraint_x}] == {constraint_y} }}, y, {{x, {start_x}, {end_x}, {dx} }}][point]
                '''
            )
        )

        self.numintegral = session.function(Global.function)
        return self.numintegral

    def data(self, startx, endx, num):
        return [([x], self.numintegral(x)) for x in np.linspace(startx, endx, num=num)]

    def wl_create_string(self, rhs):
        self.wlrhs = ""

    def wl_set_string(self, rhs):
        self.wlrhs = rhs

    def __init__(self, rhs, prototype="f(x)"):
        self.rhs = rhs
        self.prototype = prototype
        self.wl_set_string(rhs)

    def __str__(self):
        return self.prototype + " = " + self.rhs

if __name__ == '__main__':
    func = Function("3*x^2 - 7*x + 3 ") # Base function for integration 3x^2 - 7x + 3, result should be x^3 - 3.5x^2 + 3x + C
    func.wl_set_string("3*x^2 - 7*x + 3")

    basefunc = Function("x^3 + 3*x") # our "guess" seed
    try:
        func.wl_nintegrate(end_x=100.0)
        X, y = zip(*func.data(0, 100, 101))
        ybase = [x**3 + 3*x for x in np.linspace(0, 100, num=101)]
    except Exception as exc:
        print(exc)
    finally:
        session.terminate()

    default_pysr_params = dict(populations=300, model_selection="best")

    model = pysr.PySRRegressor(
        niterations=30,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[str(basefunc)],
        extra_sympy_mappings={"f": lambda x: x**3 + 3*x},
        **default_pysr_params
    )
    Xarr = np.array(list(X))
    yarr = np.array(list(y))

    print("beginning fit")
    model.fit(Xarr, yarr)

    model2 = pysr.PySRRegressor(
        niterations=30,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[""],
        **default_pysr_params
    )

    model2.fit(Xarr, yarr)

    model3 = pysr.PySRRegressor(
        niterations=30,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[""],
        **default_pysr_params
    )

    model3.fit(Xarr, yarr - np.array(ybase))

    print(model)
    print(model2)
    print(model3)
