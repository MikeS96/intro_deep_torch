import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    entropy = 0
    for yi, pi in zip(Y, P):
        entropy -= yi*np.log(pi) + (1 - yi)*np.log(1 - pi)
        
    # Y = np.float_(Y)
    # P = np.float_(P)
    # return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    
    return entropy
        
