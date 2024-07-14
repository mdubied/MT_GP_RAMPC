import numpy as np

# Generate the function W(x,y), where W is parametrised by 2 states of the system
# Generate exponents combination for monomials of 2 variables. 
# Example: monomials up to degree 2 leads to [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]
# so that it represents 1,x,y,x^2,xy,y^2
def generate_monomials(degree):
    monomials = []
    for d in range(degree + 1):
        for i in range(d + 1):
            monomials.append((d - i, i))
    return monomials

# Evaluate the a monomial expression at (x,y) based on the generated coefficients "monomials"
# Example: return evaluation of [1,x,y,x^2,xy,y^2] if the degree is 2
def evaluate_monomials(monomials, x, y):
    return [x**i * y**j for (i, j) in monomials]

# Construct a function by multiplying a matrix of coefficients with the corresponding monomial
def construct_function_coef_monomials(coef, monomial_degree):
    monomials = generate_monomials(monomial_degree)
    n_vW = len(monomials)
    n_row = coef.shape[0]
    n_col = coef.shape[1]
    
    def f(x, y):
        v_W = evaluate_monomials(monomials, x, y)
        result = np.zeros((n_row, n_col))
        for i in range(n_vW):
            result += coef[:, :, i] * v_W[i]
        return result
    
    return f
