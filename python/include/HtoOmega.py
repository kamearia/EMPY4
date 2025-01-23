from ngsolve import *
import sys
sys.path.append('C:\EMSolution\EMSolPy3\python\include')
from MatrixSolver import MatrixSolver as solver 
def HtoOmega(mesh, boundary, feOrder, H):
    fesOmega = H1(mesh, order=feOrder, definedon=mesh.Boundaries(boundary), complex=False)
    omega, psi= fesOmega.TnT()

    a = BilinearForm(fesOmega)
    a +=grad(omega).Trace()*grad(psi).Trace()*ds
    f=LinearForm(fesOmega)
    f += (grad(psi).Trace()*H)*ds
    with TaskManager():
        a.Assemble()
        f.Assemble()
    gfOmega=GridFunction(fesOmega)
    gfOmega=solver.iccg_solve(fesOmega, gfOmega, a, f.vec.FV(), tol=1.e-16, max_iter=200, accel_factor=0, complex=False) 
    return gfOmega