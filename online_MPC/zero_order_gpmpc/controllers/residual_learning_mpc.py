import os, sys, shutil
import numpy as np
import casadi as cas

import torch
import gpytorch

from acados_template import AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
from .zoro_acados_utils import *
from zero_order_gpmpc.models import ResidualModel

from time import perf_counter
from dataclasses import dataclass

@dataclass
class SolveData:
    n_iter: int
    sol_x: np.ndarray
    sol_u: np.ndarray 
    timings_total: float
    timings: dict

# removed: sim:AcadosSim, B:np.ndarray=None (needs to be directly in the residual model in our case)
#         path_json_sim="zoro_sim_solver_config.json",
class ResidualLearningMPC():
    def __init__(self, ocp:AcadosOcp,
        residual_model:ResidualModel=None, 
        use_cython=True, 
        path_json_ocp="zoro_ocp_solver_config.json",
        build_c_code=True
    ):
        """
        ocp: AcadosOcp for nominal problem
        sim: AcadosSim for nominal model
        residual_model: ResidualModel class
        """
        # optional argument
        # if B is None:
            # B = np.eye(ocp.dims.nx)
        
        # transform OCP to linear-params-model
        # self.B = B
        # self.sim = sim
        self.ocp = transform_ocp(ocp)

        # get dimensions
        self.nx = self.ocp.dims.nx
        self.nu = self.ocp.dims.nu
        self.np_nonlin = ocp.dims.np
        self.np_linmdl = self.ocp.dims.np
        self.N = self.ocp.dims.N
        # self.nw = B.shape[1]
        self.nw = ocp.dims.nx       # TODO: double check

        # allocation
        self.x_hat_all = np.zeros((self.N+1, self.nx))
        self.u_hat_all = np.zeros((self.N, self.nu))
        self.y_hat_all = np.zeros((self.N, self.nx+self.nu))
        self.residual_fun = np.zeros((self.N, self.nw))
        self.residual_jac = np.zeros((self.nw, self.N, self.nx+self.nu))
        self.p_hat_nonlin = np.array([ocp.parameter_values for _ in range(self.N)])
        self.p_hat_linmdl = np.array([self.ocp.parameter_values for _ in range(self.N)])

        self.has_residual_model = False
        if residual_model is not None:
            self.has_residual_model = True
            self.residual_model = residual_model
        
        self.setup_solve_stats()
        
        self.build_c_code_done = False
        if build_c_code:
            self.build(use_cython=use_cython)
    
    
    def build(self, 
        use_cython=False, 
        path_json_ocp="residual_lbmpc_ocp_solver_config.json"
    ):
        if use_cython:
            if build_c_code:
                AcadosOcpSolver.generate(self.ocp, json_file = path_json_ocp)
                AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
                # AcadosSimSolver.generate(self.sim, json_file = path_json_sim)
                # AcadosSimSolver.build(self.sim.code_export_directory, with_cython=True)

            self.ocp_solver = AcadosOcpSolver.create_cython_solver(path_json_ocp)
            # self.sim_solver = AcadosSimSolver.create_cython_solver(path_json_sim)
        else:
            self.ocp_solver = AcadosOcpSolver(self.ocp, json_file = path_json_ocp)
            # self.sim_solver = AcadosSimSolver(self.sim, json_file = path_json_sim)
        
        self.build_c_code_done = True
        
    def solve(self, tol_nlp=1e-5, n_iter_max=70):   # default value: 1e-6, 30
        time_total = perf_counter()
        self.init_solve_stats(n_iter_max)


        for i in range(n_iter_max):

            # store previous solution
            previous_sol_X, previous_sol_U = self.get_solution()

            # solve SQP
            time_iter = perf_counter()
            status_prep = self.preparation(i)
            status_feed = self.feedback(i)

            # ------------------- Check termination --------------------
            # check on residuals and terminate loop.
            time_check_termination = perf_counter()
            
            # self.ocp_solver.print_statistics() # encapsulates: stat = self.ocp_solver.get_stats("statistics")
            residuals = self.ocp_solver.get_residuals()
            print("residuals after ", i, "zoro outer iterations:\n", residuals)

            self.solve_stats["timings"]["check_termination"][i] += perf_counter() - time_check_termination
            self.solve_stats["timings"]["total"][i] += perf_counter() - time_iter

            # get current solution at shooting nodes
            current_sol_X, current_sol_U = self.get_solution()

            # compute relative difference with previous solution
            rel_dif_sol = np.max(np.abs(current_sol_X-previous_sol_X)/(np.abs(previous_sol_X)+1e-3))
            avg_rel_error = np.average(np.abs(current_sol_X-previous_sol_X)/(np.abs(previous_sol_X)+1e-3))
            print('Maximum relative error with last linearisation:', rel_dif_sol)
            print('Average relative error with last linearisation:', avg_rel_error)
            
            # if status_feed != 0:
            #     raise Exception('acados self.ocp_solver returned status {} in time step {}. Exiting.'.format(status_feed, i))     # TODO: get rid of comment again if using SQP_RTI

            # if max(residuals) < tol_nlp:
            #     break
            if rel_dif_sol < tol_nlp:
                break
        
        self.solve_stats["n_iter"] = i + 1
        self.solve_stats["timings_total"] = perf_counter() - time_total

    def preparation(self, i):
        # ------------------- Query nodes --------------------
        time_query_nodes = perf_counter()
        # preparation rti_phase (solve() AFTER setting params to get right Jacobians)
        # self.ocp_solver.options_set('rti_phase', 1) # TODO: comment out

        # get sensitivities for all stages
        for stage in range(self.N):
            # current stage values
            self.x_hat_all[stage,:] = self.ocp_solver.get(stage,"x")   
            self.u_hat_all[stage,:] = self.ocp_solver.get(stage,"u")   
            self.y_hat_all[stage,:] = np.hstack((self.x_hat_all[stage,:],self.u_hat_all[stage,:])).reshape((1,self.nx+self.nu))   
            # print(self.ocp_solver.get(stage,"u"))

        self.solve_stats["timings"]["query_nodes"][i] += perf_counter() - time_query_nodes
        
        # ------------------- Sensitivities --------------------
        time_get_gp_sensitivities = perf_counter()

        if self.has_residual_model:
            # time_get_gp_evaluate = perf_counter()
            # self.residual_fun = self.residual_model.evaluate(self.y_hat_all)
            # self.solve_stats["timings"]["get_gp_evaluate"][i] += perf_counter() - time_get_gp_evaluate
            # time_get_gp_jacobian = perf_counter()
            # self.residual_jac = self.residual_model.jacobian(self.y_hat_all)
            # self.solve_stats["timings"]["get_gp_jacobian"][i] += perf_counter() - time_get_gp_jacobian
            self.residual_fun, self.residual_jac = self.residual_model.value_and_jacobian(self.y_hat_all)

        self.solve_stats["timings"]["get_gp_sensitivities"][i] += perf_counter() - time_get_gp_sensitivities
        
        # ------------------- Update stages --------------------
        for stage in range(self.N):
            # set parameters (linear matrices and offset)
            # deleted: integration not needed as directly performed using the residual model
            # ------------------- Integrate --------------------
            """             time_integrate_set = perf_counter()
            self.sim_solver.set("x", self.x_hat_all[stage,:])
            self.sim_solver.set("u", self.u_hat_all[stage,:])
            self.sim_solver.set("p", self.p_hat_nonlin[stage,:])
            self.solve_stats["timings"]["integrate_set"][i] += perf_counter() - time_integrate_set

            time_integrate_acados_python = perf_counter()
            status_integrator = self.sim_solver.solve()
            self.solve_stats["timings"]["integrate_acados_python"][i] += perf_counter() - time_integrate_acados_python
            self.solve_stats["timings"]["integrate_acados"][i] += self.sim_solver.get("time_tot")

            time_integrate_get = perf_counter()
            A_nom = self.sim_solver.get("Sx")
            B_nom = self.sim_solver.get("Su")
            x_nom = self.sim_solver.get("x")
            self.solve_stats["timings"]["integrate_get"][i] += perf_counter() - time_integrate_get """

            # deleted: A_nom and B_nom - we get everything at once from residual model
            # deleted: multiplicatoin with self.B - performed directly in residual model
            # ------------------- Build linear model --------------------
            time_build_lin_model = perf_counter()

            A_total = self.residual_jac[:,stage,0:self.nx]
            # print('A')
            # print(A_total.shape)
            B_total = self.residual_jac[:,stage,self.nx:self.nx+self.nu]
            # print('B')
            # print(B_total.shape)
            f_hat = self.residual_fun[stage,:] \
                - A_total @ self.x_hat_all[stage,:] - B_total @ self.u_hat_all[stage,:]
            
            """ A_total = A_nom + self.B @ self.residual_jac[:,stage,0:self.nx]
            B_total = B_nom + self.B @ self.residual_jac[:,stage,self.nx:self.nx+self.nu]

            f_hat = x_nom + self.B @ self.residual_fun[stage,:] \
                - A_total @ self.x_hat_all[stage,:] - B_total @ self.u_hat_all[stage,:] """


            self.solve_stats["timings"]["build_lin_model"][i] += perf_counter() - time_build_lin_model
            
            # ------------------- Set sensitivities --------------------
            time_set_sensitivities_reshape = perf_counter()

            A_reshape = np.reshape(A_total,(self.nx**2),order="F")
            B_reshape = np.reshape(B_total,(self.nx*self.nu),order="F")
            
            self.solve_stats["timings"]["set_sensitivities_reshape"][i] += perf_counter() - time_set_sensitivities_reshape
            time_set_sensitivities = perf_counter()

            self.p_hat_linmdl[stage,:] = np.hstack((
                A_reshape,
                B_reshape,
                f_hat,
                self.p_hat_nonlin[stage,:]
            ))
            # print('_____')
            # print('p_hat_nonlin:')
            # print(self.p_hat_nonlin[stage,:])
            self.ocp_solver.set(stage, "p", self.p_hat_linmdl[stage,:])

            self.solve_stats["timings"]["set_sensitivities"][i] += perf_counter() - time_set_sensitivities

        # feedback rti_phase
        # self.ocp_solver.options_set('rti_phase', 1)      
        # ------------------- Phase 1 --------------------
        time_phase_one = perf_counter()
        # status = self.ocp_solver.solve()    # TODO: comment out when using SQP instead of SQP_RTI
        self.solve_stats["timings"]["phase_one"][i] += perf_counter() - time_phase_one

    def feedback(self, i):
        # ------------------- Solve QP --------------------
        time_solve_qp = perf_counter()

        # self.ocp_solver.options_set('rti_phase', 2) # TODO: comment out when using SQP instead of SQP_RTI
        status = self.ocp_solver.solve()
            
        self.solve_stats["timings"]["solve_qp"][i] += perf_counter() - time_solve_qp
        self.solve_stats["timings"]["solve_qp_acados"][i] += self.ocp_solver.get_stats("time_tot")

        return status

    def get_solution(self):
        X = np.zeros((self.N+1, self.nx))
        U = np.zeros((self.N, self.nu))

        # get data
        for i in range(self.N):
            X[i,:] = self.ocp_solver.get(i, "x")
            U[i,:] = self.ocp_solver.get(i, "u")

        X[self.N,:] = self.ocp_solver.get(self.N, "x")

        return X,U

    def print_solve_stats(self):
        n_iter = self.solve_stats["n_iter"]

        time_other = 0.0
        for key, t_arr in self.solve_stats["timings"].items():
            for i in range(n_iter):
                t_sum = np.sum(t_arr[0:n_iter])
                t_avg = t_sum / n_iter
                t_max = np.max(t_arr[0:n_iter])
                t_min = np.min(t_arr[0:n_iter])
                if key != "integrate_acados":
                    if key != "total":
                        time_other -= t_sum
                    else:
                        time_other += t_sum
            print(f"{key:20s}: {1000*t_sum:8.3f}ms ({n_iter} calls), {1000*t_avg:8.3f}/{1000*t_max:8.3f}/{1000*t_min:8.3f}ms (avg/max/min per call)")

        key = "other"
        t = time_other
        print("----------------------------------------------------------------")
        print(f"{key:20s}: {1000*t:8.3f}ms ({n_iter} calls), {1000*t/n_iter:8.3f}ms (1 call)")
    
    def init_solve_stats(self, max_iter):
        self.solve_stats = self.solve_stats_default.copy()
        for k in self.solve_stats["timings"].keys():
            self.solve_stats["timings"][k] = np.zeros((max_iter,))
    
    def get_solve_stats(self):
        X,U,P = self.get_solution()
        for k in self.solve_stats["timings"].keys():
            self.solve_stats["timings"][k] = self.solve_stats["timings"][k][0:self.solve_stats["n_iter"]]

        return SolveData(
            self.solve_stats["n_iter"], 
            X, 
            U, 
            self.solve_stats["timings_total"],
            self.solve_stats["timings"].copy()
        )

    def setup_solve_stats(self):
        # timings
        self.solve_stats_default = {
            "n_iter": 0,
            "timings_total": 0.0,
            "timings": {
                "build_lin_model": 0.0,
                "query_nodes": 0.0,
                "get_gp_evaluate": 0.0,
                "get_gp_jacobian": 0.0,
                "get_gp_sensitivities": 0.0,
                "integrate_acados": 0.0,
                "integrate_acados_python": 0.0,
                "integrate_get": 0.0,
                "integrate_set": 0.0,
                "set_sensitivities": 0.0,
                "set_sensitivities_reshape": 0.0,
                "propagate_covar": 0.0,
                "get_backoffs": 0.0,
                "get_backoffs_htj_sig": 0.0,
                "get_backoffs_htj_sig_matmul": 0.0,
                "get_backoffs_add": 0.0,
                "set_tightening": 0.0,
                "phase_one": 0.0,
                "check_termination": 0.0,
                "solve_qp": 0.0,
                "solve_qp_acados": 0.0,
                "total": 0.0,
            }
        }
        self.solve_stats = self.solve_stats_default.copy()