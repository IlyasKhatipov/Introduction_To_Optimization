import unittest


# Task 1: Bisection Method for Root-Finding
def bisection_method(func, a, b, tol):
    """
    Implements the Bisection Method to find a root of the function func within the interval [a, b].

    Parameters:
    - func: The function for which the root is to be found.
    - a (float): The lower bound of the interval.
    - b (float): The upper bound of the interval.
    - tol (float): The tolerance for convergence.

    Returns:
    - float: The approximate root of the function.
    """
    fa = func(a)
    fb = func(b)

    if fa == 0:
        return a
    if fb == 0:
        return b

    if fa * fb > 0:
        raise ValueError("The function must have different signs at a and b.")

    while abs(b - a) > tol:
        c = (a + b) / 2
        fc = func(c)

        if fc == 0 or abs(fc) < tol:
            return c
        elif fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return (a + b) / 2


# Task 2: Golden Section Method for Unimodal Function Optimization
def golden_section_method(func, a, b, tol):
    """
    Implements the Golden Section Method to find the minimum of a unimodal function within [a, b].

    Parameters:
    - func: The unimodal function to minimize.
    - a (float): The lower bound of the interval.
    - b (float): The upper bound of the interval.
    - tol (float): The tolerance for convergence.

    Returns:
    - tuple: A tuple containing the approximate xmin and f(xmin).
    """
    gr = (1 + 5 ** 0.5) / 2  # Golden ratio

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(b - a) > tol:
        fc = func(c)
        fd = func(d)

        if fc < fd:
            b = d
            d = c
            c = b - (b - a) / gr
        else:
            a = c
            c = d
            d = a + (b - a) / gr

    xmin = (a + b) / 2
    return xmin, func(xmin)


# Task 3: Gradient Ascent Method for Maximizing a Function
def gradient_ascent(func, dfunc, x0, alpha, iterations):
    """
    Implements the Gradient Ascent Method to find the maximum of a differentiable function.

    Parameters:
    - func: The function to maximize.
    - dfunc: The derivative of the function.
    - x0 (float): The initial guess.
    - alpha (float): The learning rate.
    - iterations (int): The number of iterations to perform.

    Returns:
    - tuple: A tuple containing the approximate xmax and f(xmax).
    """
    x = x0
    for i in range(iterations):
        grad = dfunc(x)
        x_new = x + alpha * grad
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {func(x_new)}")
        x = x_new
    return x, func(x)


# Unit Tests
class TestOptimizationMethods(unittest.TestCase):
    # Task 1 Tests
    def test_bisection_method_root_at_a(self):
        # Function f(x) = x^3 - 6x^2 + 11x - 6 has a root at x=1
        f1 = lambda x: x ** 3 - 6 * x ** 2 + 11 * x - 6
        root = bisection_method(f1, 1, 2, 1e-6)
        self.assertAlmostEqual(root, 1.0, places=6)

    def test_bisection_method_root_inside_interval(self):
        # Adjusted interval [1.5, 2.5] where the root is at x=2
        f1 = lambda x: x ** 3 - 6 * x ** 2 + 11 * x - 6
        root = bisection_method(f1, 1.5, 2.5, 1e-6)
        self.assertAlmostEqual(root, 2.0, places=6)

    def test_bisection_method_invalid_interval(self):
        # Interval where f(a)*f(b) > 0 without any endpoint being a root
        f1 = lambda x: x ** 3 - 6 * x ** 2 + 11 * x - 6
        with self.assertRaises(ValueError):
            bisection_method(f1, 4, 5, 1e-6)

    def test_bisection_method_endpoint_root_b(self):
        # Interval [3,4] where f(a)=0 (root at a)
        f1 = lambda x: x ** 3 - 6 * x ** 2 + 11 * x - 6
        root = bisection_method(f1, 3, 4, 1e-6)
        self.assertAlmostEqual(root, 3.0, places=6)

    # Task 2 Tests
    def test_golden_section_method_minimum(self):
        # Function f(x) = (x - 2)^2 + 3 has a minimum at x=2
        f2 = lambda x: (x - 2) ** 2 + 3
        xmin, fmin = golden_section_method(f2, 0, 5, 1e-4)
        self.assertAlmostEqual(xmin, 2.0, places=4)
        self.assertAlmostEqual(fmin, 3.0, places=4)

    # Task 3 Tests
    def test_gradient_ascent_method_maximum(self):
        # Function f(x) = -x^2 + 4x +1 has a maximum at x=2
        f3 = lambda x: -x ** 2 + 4 * x + 1
        df3 = lambda x: -2 * x + 4
        xmax, fmax = gradient_ascent(f3, df3, 0, 0.1, 100)
        self.assertAlmostEqual(xmax, 2.0, places=4)
        self.assertAlmostEqual(fmax, 5.0, places=4)


if __name__ == '__main__':
    f1 = lambda x: x ** 3 - 6 * x ** 2 + 11 * x - 6
    try:
        root = bisection_method(f1, 1, 2, 1e-6)
        print(f"Task 1: Approximate root = {root}")
    except ValueError as e:
        print(f"Task 1: {e}")

    # Task 2 Example Test
    f2 = lambda x: (x - 2) ** 2 + 3
    xmin, fmin = golden_section_method(f2, 0, 5, 1e-4)
    print(f"Task 2: Approximate xmin = {xmin}, f(xmin) = {fmin}")

    # Task 3 Example Test
    f3 = lambda x: -x ** 2 + 4 * x + 1
    df3 = lambda x: -2 * x + 4
    xmax, fmax = gradient_ascent(f3, df3, 0, 0.1, 100)
    print(f"Task 3: Approximate xmax = {xmax}, f(xmax) = {fmax}")

    # Run Unit Tests
    unittest.main(argv=[''], exit=False)
