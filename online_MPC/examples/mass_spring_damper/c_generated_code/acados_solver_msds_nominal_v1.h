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

#ifndef ACADOS_SOLVER_msds_nominal_v1_H_
#define ACADOS_SOLVER_msds_nominal_v1_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define MSDS_NOMINAL_V1_NX     2
#define MSDS_NOMINAL_V1_NZ     0
#define MSDS_NOMINAL_V1_NU     1
#define MSDS_NOMINAL_V1_NP     10
#define MSDS_NOMINAL_V1_NBX    0
#define MSDS_NOMINAL_V1_NBX0   0
#define MSDS_NOMINAL_V1_NBU    0
#define MSDS_NOMINAL_V1_NSBX   0
#define MSDS_NOMINAL_V1_NSBU   0
#define MSDS_NOMINAL_V1_NSH    0
#define MSDS_NOMINAL_V1_NSH0   0
#define MSDS_NOMINAL_V1_NSG    0
#define MSDS_NOMINAL_V1_NSPHI  0
#define MSDS_NOMINAL_V1_NSHN   0
#define MSDS_NOMINAL_V1_NSGN   0
#define MSDS_NOMINAL_V1_NSPHIN 0
#define MSDS_NOMINAL_V1_NSPHI0 0
#define MSDS_NOMINAL_V1_NSBXN  0
#define MSDS_NOMINAL_V1_NS     0
#define MSDS_NOMINAL_V1_NS0    0
#define MSDS_NOMINAL_V1_NSN    0
#define MSDS_NOMINAL_V1_NG     0
#define MSDS_NOMINAL_V1_NBXN   0
#define MSDS_NOMINAL_V1_NGN    0
#define MSDS_NOMINAL_V1_NY0    3
#define MSDS_NOMINAL_V1_NY     3
#define MSDS_NOMINAL_V1_NYN    2
#define MSDS_NOMINAL_V1_N      40
#define MSDS_NOMINAL_V1_NH     6
#define MSDS_NOMINAL_V1_NHN    6
#define MSDS_NOMINAL_V1_NH0    8
#define MSDS_NOMINAL_V1_NPHI0  0
#define MSDS_NOMINAL_V1_NPHI   0
#define MSDS_NOMINAL_V1_NPHIN  0
#define MSDS_NOMINAL_V1_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct msds_nominal_v1_solver_capsule
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

} msds_nominal_v1_solver_capsule;

ACADOS_SYMBOL_EXPORT msds_nominal_v1_solver_capsule * msds_nominal_v1_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_free_capsule(msds_nominal_v1_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_create(msds_nominal_v1_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_reset(msds_nominal_v1_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of msds_nominal_v1_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_create_with_discretization(msds_nominal_v1_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_update_time_steps(msds_nominal_v1_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_update_qp_solver_cond_N(msds_nominal_v1_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_update_params(msds_nominal_v1_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_update_params_sparse(msds_nominal_v1_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_solve(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_free(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void msds_nominal_v1_acados_print_stats(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int msds_nominal_v1_acados_custom_update(msds_nominal_v1_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *msds_nominal_v1_acados_get_nlp_in(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *msds_nominal_v1_acados_get_nlp_out(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *msds_nominal_v1_acados_get_sens_out(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *msds_nominal_v1_acados_get_nlp_solver(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *msds_nominal_v1_acados_get_nlp_config(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *msds_nominal_v1_acados_get_nlp_opts(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *msds_nominal_v1_acados_get_nlp_dims(msds_nominal_v1_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *msds_nominal_v1_acados_get_nlp_plan(msds_nominal_v1_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_msds_nominal_v1_H_
