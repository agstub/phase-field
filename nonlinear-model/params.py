# All model/numerical parameters are set here.
import numpy as np

# rheology:
A0 = 1e-32                         # Glen's law coefficient (ice softness, Pa^{-n}/s)
n = 4.0                            # Glen's law exponent
B0 = A0**(-1/n)                    # Ice hardness (Pa s^{1/n})
B = (2**((n-1.0)/(2*n)))*B0        # "2*Viscosity" constant in weak form (Pa s^{1/n})
rm2 = 1 + 1.0/n - 2.0              # Exponent in weak form: r-2
eta0 = 1e14                        # viscosity at zero deviatoric stress 
eps_v = (2*eta0/B)**(2.0/rm2)      # Flow law regularization parameter

# densities:
rho_i = 917.0                      # Density of ice
rho_w = 1020.0                     # Density of water
g = 9.81                           # Gravitational acceleration

# geometry: 
H = 200.0                          # Height of the domain
L = 30*H                           # Length of the domain
sea_level = H*(rho_i/rho_w)        # Sea level elevation

# Mesh parameters
Nx = int(L/20)                     # Number of elements in x direction
Nz = int(H/20)                     # Number of elements in z direction

# Time-stepping parameters 
t_e = (4*eta0/((rho_w-rho_i)*g*H))*(rho_w/rho_i) # characteristic time scale
t_f = 1.0*t_e                                      # final time
nt = 1000                                        # number of time steps
dt = t_f/nt                                      # timestep size
t = np.linspace(0,t_f,nt)                        # time array

# elasticity / fracture parameters
Ey = 9.5e9              # Young's modulus (Pa)
nu = 0.35               # Poisson's ratio
G = Ey/(2*(1+nu))       # shear modulus (Pa)
KI = 1e5                # fracture toughness  (Pa m^1/2)     
Gc = KI**2 / Ey         # critical fracture energy (Pa m)


# post processing grid size for interpolating FEM solutions
nxi = 101
nzi = 101
