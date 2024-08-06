# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy import linalg as npla
import scipy.linalg as scla

def calculate_moment(source_probs, source_expectations, order):
  return sum([p * expect**order for (p, expect) in zip(source_probs, source_expectations)])

def calculate_moments(source_probs, source_expectations, k):
  result = []
  for i in range(2*k + 1):
    result.append(calculate_moment(source_probs, source_expectations, i))
  return result

#################################### MATRIX PENCIL ########################################
def Vandermonde(x, k):
  """
  Returns a matrix with all 1s for the first row, x for second row,
  x^2 elementwise for the third row, etc for k rows
  """
  return np.array([np.array(x)**i for i in range(k)])

# Find probabilities of sources
def recover_source_probs(moments, x, k):
  V = Vandermonde(x, k)
  V_inv = npla.inv(V)
  return np.matmul(V_inv, moments[:k])

# First moment should be 1
def matrix_pencil(moments, k):
  A = np.ones((k, k))
  B = np.ones((k, k))
  for i in range(k):
    for j in range(k):
      B[i, j] = moments[i + j]
      A[i, j] = moments[i + j + 1]
  recovered_exps = np.real(scla.eigvals(A, b=B))
  source_probs = recover_source_probs(moments, recovered_exps, k)
  return source_probs, recovered_exps

##################################### PRONY ##########################################

# Note: edit this to implement a polynomial root finder of your choice
def root_finder(c):
  """
  Finds roots of the polynomial::

    c[0] * x**n + c[1] * x**(n-1) + ... + c[n-1]*x + c[n]
  """
  return np.roots(c)

# Main algorithm
def prony(moments, k):
  H = np.zeros((k+1, k+1))
  for i in range(k+1):
    for j in range(k+1):
      H[i, j] = moments[i+j]
  w, v = npla.eig(H)
  
  smallest_eigenvector = v.T[-1]
  coefficient_list = list(smallest_eigenvector)
  coefficient_list.reverse()
  x = root_finder(coefficient_list)
  source_probs = recover_source_probs(moments, x, k)
  return source_probs, x

############################ TESTING #############################



#k=2
#test_p = [.5, .5]
#test_exp = [-.2, .7]
#moments = calculate_moments(test_p, test_exp, k)
#print(moments)
#print("Matrix Pencil:")
#print(matrix_pencil(moments, k))
#print("Prony:")
#print(prony(moments, k))