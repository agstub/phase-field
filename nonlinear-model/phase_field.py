# This file contains the functions needed for solving the phase field
# fracture problem 
from dolfinx.fem import Function, FunctionSpace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from params import Gc, L, Nx
from ufl import TestFunction, dx, grad, inner


def weak_form(c,c_t,E):
    # Weak form residual of the phase field problem
    l0 = 10 # L/Nx
    F = (l0**2)*inner(grad(c),grad(c_t))*dx + c*c_t*dx
    F -= 2*(l0/Gc)*(1-c)*E*c_t*dx
    return F

def phase_solve(domain,E):
        # solve for phase field variable given the strain
        # energy density E

        # Define function space for phase field variable
        V = FunctionSpace(domain, ("CG", 1))

        # Define variational problem 
        c = Function(V)
        c_t = TestFunction(V)
      
        # Define weak form
        F = weak_form(c,c_t,E)

        # Solve for phase-field variable c
        problem = NonlinearProblem(F, c, bcs=[])
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        n, converged = solver.solve(c)

        return c