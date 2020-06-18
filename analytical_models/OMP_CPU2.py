"""
    OMP CPU2.py
        by Tim Müller
    
    This file implements the analytical model for the second variation of the
    OpenMP CPU-optimised implementations.
"""


import sequential


# Directly implement sequential
def predict(sample_parameters, machine_parameters):
    return sequential.predict(sample_parameters, machine_parameters)
