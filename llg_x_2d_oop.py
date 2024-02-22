#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:46:50 2024

@author: mnv
"""

from mpi4py import MPI
from dolfinx import mesh, default_scalar_type
from dolfinx import fem, plot
from dolfinx.fem.petsc import (NonlinearProblem, LinearProblem, create_matrix, create_vector,
                               assemble_matrix, assemble_vector, apply_lifting, set_bc)
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from ufl import (FiniteElement, VectorElement, sin, cos, atan, exp, sqrt, SpatialCoordinate, 
                 as_vector, dx, grad, inner, dot, div, cross,
                 TrialFunction, TestFunction, derivative, Coefficient, replace, FacetNormal, Measure)
from petsc4py import PETSc
#import matplotlib.pyplot as plt
#import pyvista
import numpy as np

class llg2_solver:
    """Solver for 2d LLG problem"""
    
    def __init__(self, Lx, Ly, mesh_2d):
        """Initialize with mesh, construct Function Space, trial function, test function"""
        
        self.Lx = Lx
        self.Ly = Ly
        self.mesh_2d = mesh_2d
        
        self.coords = SpatialCoordinate(self.mesh_2d)
        
        v_element = VectorElement("CG", mesh_2d.ufl_cell(), degree = 1, dim = 3)
        f_element = FiniteElement("CG", mesh_2d.ufl_cell(), degree = 1)

        self.FS = fem.FunctionSpace(mesh_2d, v_element)
        self.FS_1d = fem.FunctionSpace(mesh_2d, f_element)
        
        self.v = fem.Function(self.FS)
        self.w = TestFunction(self.FS)
        
        self.vp = TrialFunction(self.FS_1d)
        self.wp = TestFunction(self.FS_1d)
        
        self.pot = fem.Function(self.FS_1d)
        
        # can be chahged by setting later
        self.h_ext = None
    
    def norm_vec(self, u):
        v = u.copy()
        N = u.x.array.size//3
        v_vec = np.reshape(v.x.array, (N, 3))
        for i in range(0, N, 1):
            #v_vec[i,0] = 0
            norm = np.sqrt(v_vec[i,0]**2 + v_vec[i,1]**2 + v_vec[i,2]**2)
            v_vec[i] = v_vec[i]/norm
        
        norm_vec = np.reshape(v_vec, 3*N)
        v.x.array[:] = norm_vec
        return v
    
    def demag_pot(self):
        
        with self.b_p.localForm() as loc:
            loc.set(0)
        
        assemble_vector(self.b_p, self.L_p)
        #apply_lifting(self.b_p, [self.a_p], [self.bc_p])
        self.b_p.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        #set_bc(self.b_p, self.bc_p)
        self.solver_p.solve(self.b_p, self.pot.vector)
        self.pot.x.scatter_forward()
        
        # problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        # self.pot = problem.solve()
        # self.pot.x.scatter_forward()
        
        #return pot
    
    def set_in_cond(self, m_in_expr, in_type='new', path_to_sol = None):
        """Set initial condition and boundary condition from fenics expression"""
    
        # if in_type == 'old':
        #     hdf_m_old = fen.HDF5File(self.mesh.mpi_comm(), path_to_sol, 'r')
        #     self.in_cond = fen.project(m_in_expr, self.FS)
        #     self.m = fen.Function(self.FS)
        #     hdf_m_old.read(self.m, "/m_field")
        #     hdf_m_old.close()
        #     self.BC = fen.DirichletBC(self.FS, self.m, boundary)
        # elif in_type == 'new':
        #     self.in_cond = fen.project(m_in_expr, self.FS)
        #     self.m = fen.project(m_in_expr, self.FS)
        #     self.BC = fen.DirichletBC(self.FS, m_in_expr, boundary)
        # else:
        #     raise NotImplementedError()
        
        self.ub = fem.Function(self.FS)
        self.ub.interpolate(m_in_expr)
        
        self.m = fem.Function(self.FS, name = 'm')
        self.m.interpolate(m_in_expr)
        
        self.m_init = fem.Function(self.FS, name = 'm_init')
        self.m_init.name = 'm_init'
        self.m_init.interpolate(m_in_expr)
        
        self.tdim = mesh_2d.topology.dim
        self.fdim = self.tdim - 1
        mesh_2d.topology.create_connectivity(self.fdim, self.tdim)
        self.boundary_facets = mesh.exterior_facet_indices(mesh_2d.topology)

        self.boundary_dofs = fem.locate_dofs_topological(self.FS, self.fdim, self.boundary_facets)
        
        def boundary_D(x):
            return np.logical_or(np.isclose(x[1], -self.Ly/2), np.isclose(x[1], self.Ly/2))


        self.dofs_D = fem.locate_dofs_geometrical(self.FS_1d, boundary_D)
        
        self.bc = fem.dirichletbc(self.ub, self.boundary_dofs)
        
        
        self.bc_p = [fem.dirichletbc(PETSc.ScalarType(0), self.dofs_D, self.FS_1d)]
        
        self.T = 1.
    
    def h_rest(self):
        m1, m2, m3 = self.m.split()
        e1, e2, e3 = self.e_v.split()
        dedz_1, dedz_2, dedz_3 = self.de_dz_v.split()
        
        oo = fem.Constant(mesh_2d, PETSc.ScalarType(0))
        vec = as_vector((-self.p*(2*e1*m1.dx(0) + 2*e2*m2.dx(0) + 2*e3*m3.dx(0) + m1*e1.dx(0) + m2*e2.dx(0) + m3*e3.dx(0) + m1*e1.dx(0) + m2*e1.dx(1) + m3*dedz_1), \
                     -self.p*(2*e1*m1.dx(1) + 2*e2*m2.dx(1) + 2*e3*m3.dx(1) + m1*e1.dx(1) + m2*e2.dx(1) + m3*e3.dx(1) + m1*e2.dx(0) + m2*e2.dx(1) + m3*dedz_2), \
                          -self.p*(m1*e3.dx(0) + m2*e3.dx(1) + m3*dedz_3 + m1*dedz_1 + m2*dedz_2 + m3*dedz_3)))
        
        self.demag_vec = as_vector((-self.pot.dx(0), -self.pot.dx(1), oo))
        self.demag_vec_func = fem.Function(self.FS)
        demag_vec_func_expr = fem.Expression(self.demag_vec, self.FS.element.interpolation_points())
        self.demag_vec_func.interpolate(demag_vec_func_expr)
        
        vec_total = as_vector((oo, oo, m3)) + vec
        
        if self.h_ext is not None:
            vec_total += self.h_ext
        return  vec_total + self.Ms**2/self.kku/2*self.demag_vec
    
    def dot_v(self, m, mm, w):
        mm1, mm2, mm3 = m.split()
        e1, e2, e3 = self.e_v.split()
        oo = fem.Constant(mesh_2d, PETSc.ScalarType(0))
        m_2d = as_vector(mm1,mm2)
        expr = dot(grad(cross(w,m)[0]),grad(mm1)  + 2*self.p*e1*m_2d) + \
            dot(grad(cross(w,m)[1]),grad(mm2)  + 2*self.p*e2*m_2d) + \
                dot(grad(cross(w,m)[2]),grad(mm3)  + 2*self.p*e3*m_2d)
        return expr
    
    def set_params(self, alpha = 1, kku = 1000, A_ex = 9.5*10**(-8), Ms = 4, pin = True):
        """Set parameters"""
        
        self.alpha = alpha
        self.kku = kku
        self.Ms = Ms
        self.A_ex = A_ex
        self.dw_width = np.sqrt(A_ex/kku)
    
    def set_comp_params(self, dt, N_f, route_0 = '/media/mnv/T7/graphs'):

        self.dt = dt
        self.N_f = N_f
    
        self.route_0 = route_0
        
    def set_h_ext(self, h_x = 0.5, h_y = 0, h_z = 0):
        """
        Set external magnetic field

        Parameters
        ----------
        h_x : TYPE, optional
            DESCRIPTION. The default is 0.
        h_y : TYPE, optional
            DESCRIPTION. The default is 0.
        h_z : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        
        factor = self.Ms**2/2/self.kku
        #self.h_ext = factor*fen.project(fen.as_vector((h_x, h_y, h_z)), self.FS)
        
        self.h_ext = factor*as_vector((h_x, h_y, h_z))
    
    def set_F(self):
        
        F_Pi = fem.Constant(self.mesh_2d, PETSc.ScalarType(4*np.pi))
        
        n = FacetNormal(self.mesh_2d)
        ds = Measure("ds", domain=self.mesh_2d)
        
        m1, m2, m3 = self.m.split()
        m_2d = as_vector((m1,m2))
        
        ub1, ub2, ub3 = self.ub.split()
        ub_2d = as_vector((ub1,ub2))
        
        self.a_p = fem.form(dot(grad(self.wp),grad(self.vp))*dx)
        self.L_p = fem.form(-F_Pi*self.wp*dot(ub_2d,n)*ds + F_Pi*dot(m_2d, grad(self.wp))*dx)
        
        A = assemble_matrix(self.a_p, bcs=self.bc_p)
        A.assemble()
        self.b_p = create_vector(self.L_p)
        
        self.solver_p = PETSc.KSP().create(self.mesh_2d.comm)
        self.solver_p.setOperators(A)
        self.solver_p.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver_p.getPC()
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")
        
        self.F = dot(self.w,(self.v-self.m)/self.dt-self.alpha*cross(self.v,(self.v-self.m)/self.dt))*dx + dot(self.w,cross(self.v, self.h_rest()))*dx - self.dot_v(self.v,self.v,self.w)*dx #+ dot(w,cross(m,dmdn(m,n)))*ds + 2*pp*dot(w,cross(m,e_f))*dot(to_2d(m),n)*ds
        J = derivative(self.F, self.v)
        self.jacobian = fem.form(J)
        
    
    def e_field_from_ps(self, x0 = 0, y0 = 0, r0 = 0.00002, U0 = 2*10/3/50, gamma_me = 1E-6):
        
        self.p = gamma_me*U0/r0/(2*np.sqrt(self.A_ex*self.kku))
        
        x = SpatialCoordinate(self.mesh_2d)
        
        d = self.dw_width
        
        # zc = Coefficient(self.FS_1d)
        
        # pot_expr = 1/sqrt((d*r0-d*zc)**2+((d*x[0]-d*x0)**2 + (d*x[1]-d*y0)**2))
        
        # E1 = -grad(pot_expr)[0]/d
        # dE1_dz = derivative(E1,zc)
        # E1 = replace(E1, {zc:0.0})
        # dE1_dz = replace(dE1_dz, {zc:0.0})
        
        # E2 = -grad(pot_expr)[1]/d
        # dE2_dz = derivative(E2,zc)
        # E2 = replace(E2, {zc:0.0})
        # dE2_dz = replace(dE2_dz, {zc:0.0})
        
        # E3 = -derivative(pot_expr, zc)
        # dE3_dz = derivative(E3,zc)
        # E3 = replace(E3, {zc:0.0})
        # dE3_dz = replace(dE3_dz, {zc:0.0})
        
        ksi_0 = r0/d
        
        r = sqrt((ksi_0)**2+((x[0]-x0)**2 + (x[1]-y0)**2))
        
        E1 = (r0/d)**2*(x[0]-x0)/r**3
        
        E2 = (r0/d)**2*(x[1]-y0)/r**3
        
        E3 = (r0/d)**2*(-ksi_0)/r**3
        
        dE1_dz = (r0/d)**2*(-3)*(x[0]-x0)*(-ksi_0)/r**5
        
        dE2_dz = (r0/d)**2*(-3)*(x[1]-x0)*(-ksi_0)/r**5
        
        dE3_dz = (r0/d)**2*(1/r**3+(-3)*(-ksi_0)*(-ksi_0)/r**5)
        
        e_v_expr = fem.Expression(as_vector((E1,E2,E3)), self.FS.element.interpolation_points())
        self.e_v = fem.Function(self.FS, name = 'e')
        self.e_v.interpolate(e_v_expr)
        
        de_dz_v_expr = fem.Expression(as_vector((dE1_dz,dE2_dz,dE3_dz)), self.FS.element.interpolation_points())
        self.de_dz_v = fem.Function(self.FS, name = 'de_dz')
        self.de_dz_v.interpolate(de_dz_v_expr)
        
        print("ME parameter p = ", self.p)
    
    def custom_newton_solver(self, v, jacobian, residual, mesh):
        
        residual = fem.form(self.F)
        
        A = create_matrix(jacobian)
        L = create_vector(residual)
        
        FS = v.function_space
        
        newton_solver = PETSc.KSP().create(mesh.comm)
        newton_solver.setOperators(A)
        delta_x = fem.Function(FS)
        
        i = 0
        #coords = FS.tabulate_dof_coordinates()[:, 0]
        #sort_order = np.argsort(coords)
        max_iterations = 25
        #solutions = np.zeros((max_iterations + 1, len(coords)))
        #solutions[0] = v.x.array[sort_order]
        
        while i <= max_iterations:
            # Assemble Jacobian and residual
            with L.localForm() as loc_L:
                loc_L.set(0)
            A.zeroEntries()
            assemble_matrix(A, jacobian)
            A.assemble()
            assemble_vector(L, residual)
            L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

            # Scale residual by -1
            L.scale(-1)
            L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

            # Solve linear problem
            newton_solver.solve(L, delta_x.vector)
            delta_x.x.scatter_forward()
            # Update u_{i+1} = u_i + delta x_i
            v.x.array[:] += delta_x.x.array
            i += 1

            # Compute norm of update
            correction_norm = delta_x.vector.norm(0)
            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < 1e-10:
                break
            #solutions[i, :] = v.x.array[sort_order]
        return v
    
    def solve(self, each_idx_write = 10):
        
        problem = NonlinearProblem(self.F, self.v, [self.bc])

        # Set Newton solver options
        solver = NewtonSolver(mesh_2d.comm, problem)
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.max_it = 100
        solver.report = True
        solver.convergence_criterion = "incremental"

        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        opts[f"{option_prefix}pc_type"] = "none"
        #opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        ksp.setFromOptions()
        
        m_init_file = XDMFFile(mesh_2d.comm, self.route_0 + "m_init.xdmf", "w")
        m_init_file.write_mesh(self.mesh_2d)
        m_init_file.write_function(self.m_init, self.T)
        m_init_file.close()
        
        m_file = XDMFFile(mesh_2d.comm, self.route_0 + "m.xdmf", "w")
        m_file.write_mesh(self.mesh_2d)
        #m_file.write_function(self.m, self.T)
        
        deriv_file = XDMFFile(mesh_2d.comm, self.route_0 + "derivative.xdmf", "w")
        deriv_file.write_mesh(self.mesh_2d)
        
        Pol_file = XDMFFile(mesh_2d.comm, self.route_0 + "Pol.xdmf", "w")
        Pol_file.write_mesh(self.mesh_2d)
        
        demag_field_file = XDMFFile(mesh_2d.comm, self.route_0 + "demag_field.xdmf", "w")
        demag_field_file.write_mesh(self.mesh_2d)
        
        
        e_file = XDMFFile(mesh_2d.comm, self.route_0 + "e.xdmf", "w")
        e_file.write_mesh(self.mesh_2d)
        e_file.write_function(self.e_v, self.T)
        e_file.close()
        
        de_dz_file = XDMFFile(mesh_2d.comm, self.route_0 + "de_dz.xdmf", "w")
        de_dz_file.write_mesh(self.mesh_2d)
        de_dz_file.write_function(self.de_dz_v, self.T)
        de_dz_file.close()
        
        derivative_func = fem.Function(self.FS, name = 'derivative')
        Pol_func = fem.Function(self.FS, name = 'Pol')
        
        oo = fem.Constant(mesh_2d, PETSc.ScalarType(0))

        for n in range(0, self.N_f):
            self.demag_pot()
            demag_vec_func_expr = fem.Expression(self.demag_vec, self.FS.element.interpolation_points())
            self.demag_vec_func.interpolate(demag_vec_func_expr)
            
            num_its, converged = solver.solve(self.v)
            assert (converged)
            
            self.v.x.array[:] = self.norm_vec(self.v).x.array
            
            self.v.x.scatter_forward()
            
            derivative_func.x.array[:] = self.v.x.array[:] - self.m.x.array[:]
            
            comm = self.mesh_2d.comm
            derivative_int = fem.form(derivative_func**2 * dx)
            derivative_avg = np.sqrt(comm.allreduce(fem.assemble_scalar(derivative_int), MPI.SUM))/(self.Lx*self.Ly*self.dt)
            
            if comm.rank == 0:
                print(f"L2-derivative: {derivative_avg:.8e}")
            #v = norm_vec(v)
            self.m.x.array[:] = self.v.x.array
            
            
            v1, v2, v3 = self.v.split()
            
            Pol_exp = fem.Expression(self.v*(v1.dx(0) + v2.dx(1)) - v1*self.v.dx(0) - v2*self.v.dx(1), self.FS.element.interpolation_points())
            Pol_func.interpolate(Pol_exp)
            
            # Pol1_int = fem.form(Pol_func[0] * dx)
            # Pol2_int = fem.form(Pol_func[1] * dx)
            # Pol3_int = fem.form(Pol_func[2] * dx)
            
            # Pol1_avg = comm.allreduce(fem.assemble_vector(Pol1_int), MPI.SUM)/(self.Lx*self.Ly)
            # Pol2_avg = comm.allreduce(fem.assemble_vector(Pol2_int), MPI.SUM)/(self.Lx*self.Ly)
            # Pol3_avg = comm.allreduce(fem.assemble_vector(Pol3_int), MPI.SUM)/(self.Lx*self.Ly)
            
            if (n%each_idx_write == 0):
                m_file.write_function(self.m, self.T)
                deriv_file.write_function(derivative_func, self.T)
                Pol_file.write_function(Pol_func, self.T)
                demag_field_file.write_function(self.demag_vec_func, self.T)
            
            self.T += self.dt
            print(f"Time step {n}, Number of iterations {num_its}")

        m_file.close()
        deriv_file.close()
        Pol_file.close()
        demag_field_file.close()


Lx = 10
Ly = 10
mesh_2d = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-Lx/2, -Ly/2]), np.array([Lx/2, Ly/2])],
                               [16*Lx, 16*Ly], mesh.CellType.triangle)

llg_solver = llg2_solver(Lx, Ly, mesh_2d)

x = SpatialCoordinate(mesh_2d)
ub_expr = fem.Expression(as_vector((sin(2*atan(exp(x[1]))), 0, cos(2*atan(exp(x[1]))))), llg_solver.FS.element.interpolation_points())

llg_solver.set_in_cond(ub_expr, in_type='new', path_to_sol = None)

llg_solver.set_params(alpha = 1E-4, kku = 1000, A_ex = 9.5*10**(-8), Ms = 4, pin = True)
#llg_solver.set_h_ext(h_x = 50)

dt = 2*0.0001
N_f = 4000

llg_solver.set_comp_params(dt, N_f, route_0 = '/home/mnv/llg_nl/results/graphs/')

llg_solver.e_field_from_ps(x0 = 0, y0 = 0.2, r0 = 0.000001, U0 = 2.5*2*10/3/50/10, gamma_me = 1E-6)

llg_solver.set_F()

llg_solver.solve(each_idx_write = 5)
