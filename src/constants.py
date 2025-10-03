import numpy as np 

c = 3e8  # Speed of light in m/s
f = 28e9 # Carrier frequency of 28 GHz
lambda_ = c / f # Wave-length
k = 2 * np.pi / lambda_  # Wave number
omega = 2 * np.pi * f # Radian frequency
P_t_dBm = 30  # Transmit power (dBm)
P_t_w = 10 ** ((P_t_dBm - 30) / 10)  # Transmit power in Watts
epsilon0 = 8.854e-12  # Permittivity of free space (F/m)
mu0 = 1.25663706e-6  # Permeability of free space (H/m)
sigma0 = 0 # Conductivity of free space
mu = mu0 # Concrete permeability
epsilon = 5.31*epsilon0 # concrete permittivity
sigma = 0.626 # conductivity of concrete at 28 GHz from literature
eta = np.sqrt(mu0/epsilon0) # impedance of free space
Zw = np.sqrt((1j * omega * mu)/(sigma + (1j * omega * epsilon))) # Impedance of building walls at 28 GHz
epsilon_g = 3.0*epsilon0
sigma_g = 0.04967
Zg = np.sqrt((1j * omega * mu)/(sigma_g + (1j * omega * epsilon_g))) # Impedance of ground at 28 GHz
WALL_HEIGHT_TOLERANCE = 1e-6
