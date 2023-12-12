#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing ice-shelf response to 
# sub-ice-shelf melting or freezing anomalies. The code relies on FEniCSx-see README
#------------------------------------------------------------------------------------

import numpy as np
from dolfinx.fem import Expression,Function,FunctionSpace
from dolfinx.mesh import create_rectangle, CellType
from mesh_routine import get_surfaces, move_mesh
from mpi4py import MPI
from params import H, L, Nx, Nz, nxi,nzi, nt, t
from stokes import stokes_solve
from phase_field import phase_solve
from post_process import interp,get_energy, surfaces,phase

def Max(f1,f2):
     # pointwise max function
     return 0.5*(f1+f2 + ((f1-f2)**2)**0.5)

def solve(a,m):

    # generate mesh
    p0 = [-L/2.0,0.0]
    p1 = [L/2.0,H]
    domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [Nx, Nz],cell_type=CellType.quadrilateral)

    # Define arrays for saving surfaces
    h_i,s_i,x = get_surfaces(domain)
    nx = x.size
    h = np.zeros((nx,nt))      # upper surface
    s = np.zeros((nx,nt))      # lower surface

    C = np.zeros((nxi,nzi,nt)) # phase field 
    X = np.zeros((nxi,nzi,nt)) # grid points - x coordinate
    Z = np.zeros((nxi,nzi,nt)) # grid points - z coordinate

    # intialize phase field variables
    V = FunctionSpace(domain, ("CG", 1))
    c = Function(V)
    E_prev = Function(V)

    # # Begin time stepping
    for i in range(nt):

        print('Iteration '+str(i+1)+' out of '+str(nt)+' \r',end='')

        t_i = t[i]

        # Solve the Stoke problem for w = (u,p)
        sol = stokes_solve(c,domain)

        # get the strain energy density
        E = get_energy(sol,domain)

        # enforce non-reversibility condition
        E_max = Max(E,E_prev)

        # solve phase field problem for c
        c = phase_solve(domain,E_max)

        # interpolate and save phase-field solution
        c_i = c.x.array
        x_i = domain.geometry.x[:,0]
        z_i = domain.geometry.x[:,1]

        C_i,X_i,Z_i = interp(h_i,s_i,x,x_i,z_i,c_i)
        C[:,:,i] = C_i
        X[:,:,i] = X_i
        Z[:,:,i] = Z_i

        # move the mesh according to the velocity solution,
        # surface accumulation, and basal melting
        domain = move_mesh(sol,domain,t_i,a,m)
       
        # save upper and lower surfaces of ice shelf
        h_i,s_i,x = get_surfaces(domain)
        h[:,i] = h_i
        s[:,i] = s_i

        # set previous strain energy density to enforce 
        # irreversibility condition
        V = FunctionSpace(domain, ("CG", 1))
        E_prev = Function(V)
        E_prev.interpolate(Expression(E_max, V.element.interpolation_points()))
    
    # bundle surfaces and phase-field solution
    geometry = surfaces(h,s,x)
    phase_field =  phase(C,X,Z)

    return geometry, phase_field