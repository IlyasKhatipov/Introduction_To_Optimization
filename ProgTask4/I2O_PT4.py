# Task 1: Bisection Method for Root-Finding
def bisection_method(func, a, b, tol):
    if func(a) * func(b) >= 0:
        raise ValueError("The function must have different signs at a and b.")

    while abs(b - a) > tol:
        c = (a + b) / 2
        if abs(func(c)) < tol:
            return c
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2

# Function for Task 1
f1 = lambda x: x**3 - 6*x**2 + 11*x - 6

# Example Test for Task 1
a, b, tol = 1, 2, 1e-6
root = bisection_method(f1, a, b, tol)
print(f"Task 1: Approximate root = {root}")

# Task 2: Golden Section Method for Unimodal Function Optimization
def golden_section_method(func, a, b, tol):
    gr = (1 + 5 ** 0.5) / 2  # Golden ratio

    while abs(b - a) > tol:
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        if func(c) < func(d):
            b = d
        else:
            a = c

    xmin = (a + b) / 2
    return xmin, func(xmin)

# Function for Task 2
f2 = lambda x: (x - 2)**2 + 3

# Example Test for Task 2
a, b, tol = 0, 5, 1e-4
xmin, fmin = golden_section_method(f2, a, b, tol)
print(f"Task 2: Approximate xmin = {xmin}, f(xmin) = {fmin}")

# Task 3: Gradient Ascent Method for Maximizing a Function
def gradient_ascent(func, dfunc, x0, alpha, iterations):
    x = x0
    for _ in range(iterations):
        grad = dfunc(x)
        x += alpha * grad

    return x, func(x)

# Function and its derivative for Task 3
f3 = lambda x: -x**2 + 4*x + 1
df3 = lambda x: -2*x + 4

# Example Test for Task 3
x0, alpha, iterations = 0, 0.1, 100
xmax, fmax = gradient_ascent(f3, df3, x0, alpha, iterations)
print(f"Task 3: Approximate xmax = {xmax}, f(xmax) = {fmax}")
