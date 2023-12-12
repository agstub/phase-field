import numpy as np
from dolfinx.fem import Expression, Function, FunctionSpace
from params import Ey,G, H, nxi, nzi
from scipy.interpolate import griddata, interp1d
from stokes import eta
from ufl import grad, inner, sym


def interp(h,s,x,X,Z,E):
    # interpolate a function ("E") on deformed grid
    # for plotting
    z_g = Z/H
    x_g = X/H
    x0 = x/H

    he = h/H-1
    se = s/H

    h_int = interp1d(x0,he)
    s_int = interp1d(x0,se)

    z_d = (1-z_g)*s_int(x_g) + z_g*(h_int(x_g)+1)

    points = (x_g,z_d)

    xi = np.linspace(x_g.min(),x_g.max(),num=nxi)
    zi = np.linspace(z_d.min(),z_d.max(),num=nzi)

    Xi,Zi = np.meshgrid(xi,zi)
    points_i = (Xi,Zi)
    Ec = griddata(points=points,values=E, xi=points_i,fill_value=0)
    return Ec, Xi, Zi


def get_energy(sol,domain):
    # get elastic strain energy density
    V = FunctionSpace(domain, ("CG", 1))
    strain_rate = sym(grad(sol.sub(0)))
    strain = (eta(sol.sub(0))/Ey)*strain_rate
    E = Function(V)
    E_expr = G*inner(strain,strain)
    E.interpolate(Expression(E_expr, V.element.interpolation_points()))
    return E


class surfaces:
    # object to store upper/lower surfaces of ice shelf
    def __init__(self, h, s, x):
        self.h = h
        self.s = s
        self.x = x

class phase:
    # object to store phase-field solution and grid points
    def __init__(self, C, X, Z):
        self.C = C
        self.X = X
        self.Z = Z