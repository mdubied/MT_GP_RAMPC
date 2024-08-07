/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_quadrotor_RAMPC_v2_H_
#define ACADOS_SOLVER_quadrotor_RAMPC_v2_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define QUADROTOR_RAMPC_V2_NX     7
#define QUADROTOR_RAMPC_V2_NZ     0
#define QUADROTOR_RAMPC_V2_NU     10
#define QUADROTOR_RAMPC_V2_NP     132
#define QUADROTOR_RAMPC_V2_NBX    0
#define QUADROTOR_RAMPC_V2_NBX0   0
#define QUADROTOR_RAMPC_V2_NBU    0
#define QUADROTOR_RAMPC_V2_NSBX   0
#define QUADROTOR_RAMPC_V2_NSBU   0
#define QUADROTOR_RAMPC_V2_NSH    27
#define QUADROTOR_RAMPC_V2_NSH0   35
#define QUADROTOR_RAMPC_V2_NSG    0
#define QUADROTOR_RAMPC_V2_NSPHI  0
#define QUADROTOR_RAMPC_V2_NSHN   22
#define QUADROTOR_RAMPC_V2_NSGN   0
#define QUADROTOR_RAMPC_V2_NSPHIN 0
#define QUADROTOR_RAMPC_V2_NSPHI0 0
#define QUADROTOR_RAMPC_V2_NSBXN  0
#define QUADROTOR_RAMPC_V2_NS     27
#define QUADROTOR_RAMPC_V2_NS0    35
#define QUADROTOR_RAMPC_V2_NSN    22
#define QUADROTOR_RAMPC_V2_NG     0
#define QUADROTOR_RAMPC_V2_NBXN   0
#define QUADROTOR_RAMPC_V2_NGN    0
#define QUADROTOR_RAMPC_V2_NY0    17
#define QUADROTOR_RAMPC_V2_NY     17
#define QUADROTOR_RAMPC_V2_NYN    7
#define QUADROTOR_RAMPC_V2_N      240
#define QUADROTOR_RAMPC_V2_NH     27
#define QUADROTOR_RAMPC_V2_NHN    22
#define QUADROTOR_RAMPC_V2_NH0    35
#define QUADROTOR_RAMPC_V2_NPHI0  0
#define QUADROTOR_RAMPC_V2_NPHI   0
#define QUADROTOR_RAMPC_V2_NPHIN  0
#define QUADROTOR_RAMPC_V2_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct quadrotor_RAMPC_v2_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *discr_dyn_phi_fun;
    external_function_param_casadi *discr_dyn_phi_fun_jac_ut_xt;


    // cost






    // constraints
    external_function_param_casadi *nl_constr_h_fun_jac;
    external_function_param_casadi *nl_constr_h_fun;



    external_function_param_casadi nl_constr_h_0_fun_jac;
    external_function_param_casadi nl_constr_h_0_fun;



    external_function_param_casadi nl_constr_h_e_fun_jac;
    external_function_param_casadi nl_constr_h_e_fun;

} quadrotor_RAMPC_v2_solver_capsule;

ACADOS_SYMBOL_EXPORT quadrotor_RAMPC_v2_solver_capsule * quadrotor_RAMPC_v2_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_free_capsule(quadrotor_RAMPC_v2_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_create(quadrotor_RAMPC_v2_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_reset(quadrotor_RAMPC_v2_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of quadrotor_RAMPC_v2_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_create_with_discretization(quadrotor_RAMPC_v2_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_update_time_steps(quadrotor_RAMPC_v2_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_update_qp_solver_cond_N(quadrotor_RAMPC_v2_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_update_params(quadrotor_RAMPC_v2_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_update_params_sparse(quadrotor_RAMPC_v2_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_solve(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_free(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void quadrotor_RAMPC_v2_acados_print_stats(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int quadrotor_RAMPC_v2_acados_custom_update(quadrotor_RAMPC_v2_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *quadrotor_RAMPC_v2_acados_get_nlp_in(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *quadrotor_RAMPC_v2_acados_get_nlp_out(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *quadrotor_RAMPC_v2_acados_get_sens_out(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *quadrotor_RAMPC_v2_acados_get_nlp_solver(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *quadrotor_RAMPC_v2_acados_get_nlp_config(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *quadrotor_RAMPC_v2_acados_get_nlp_opts(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *quadrotor_RAMPC_v2_acados_get_nlp_dims(quadrotor_RAMPC_v2_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *quadrotor_RAMPC_v2_acados_get_nlp_plan(quadrotor_RAMPC_v2_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_quadrotor_RAMPC_v2_H_
