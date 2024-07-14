/* This file was automatically generated by CasADi 3.6.4.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) msds_RAMPC_v1_dyn_disc_phi_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s2[51] = {47, 1, 0, 47, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46};

/* msds_RAMPC_v1_dyn_disc_phi_fun:(i0[3],i1[11],i2[47])->(o0[3]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[2]? arg[2][0] : 0;
  a1=arg[0]? arg[0][0] : 0;
  a0=(a0*a1);
  a2=arg[2]? arg[2][3] : 0;
  a3=arg[0]? arg[0][1] : 0;
  a2=(a2*a3);
  a0=(a0+a2);
  a2=arg[2]? arg[2][6] : 0;
  a4=arg[0]? arg[0][2] : 0;
  a2=(a2*a4);
  a0=(a0+a2);
  a2=arg[2]? arg[2][9] : 0;
  a5=arg[1]? arg[1][0] : 0;
  a2=(a2*a5);
  a6=arg[2]? arg[2][12] : 0;
  a7=arg[1]? arg[1][1] : 0;
  a6=(a6*a7);
  a2=(a2+a6);
  a6=arg[2]? arg[2][15] : 0;
  a8=arg[1]? arg[1][2] : 0;
  a6=(a6*a8);
  a2=(a2+a6);
  a6=arg[2]? arg[2][18] : 0;
  a9=arg[1]? arg[1][3] : 0;
  a6=(a6*a9);
  a2=(a2+a6);
  a6=arg[2]? arg[2][21] : 0;
  a10=arg[1]? arg[1][4] : 0;
  a6=(a6*a10);
  a2=(a2+a6);
  a6=arg[2]? arg[2][24] : 0;
  a11=arg[1]? arg[1][5] : 0;
  a6=(a6*a11);
  a2=(a2+a6);
  a6=arg[2]? arg[2][27] : 0;
  a12=arg[1]? arg[1][6] : 0;
  a6=(a6*a12);
  a2=(a2+a6);
  a6=arg[2]? arg[2][30] : 0;
  a13=arg[1]? arg[1][7] : 0;
  a6=(a6*a13);
  a2=(a2+a6);
  a6=arg[2]? arg[2][33] : 0;
  a14=arg[1]? arg[1][8] : 0;
  a6=(a6*a14);
  a2=(a2+a6);
  a6=arg[2]? arg[2][36] : 0;
  a15=arg[1]? arg[1][9] : 0;
  a6=(a6*a15);
  a2=(a2+a6);
  a6=arg[2]? arg[2][39] : 0;
  a16=arg[1]? arg[1][10] : 0;
  a6=(a6*a16);
  a2=(a2+a6);
  a0=(a0+a2);
  a2=arg[2]? arg[2][42] : 0;
  a0=(a0+a2);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[2]? arg[2][1] : 0;
  a0=(a0*a1);
  a2=arg[2]? arg[2][4] : 0;
  a2=(a2*a3);
  a0=(a0+a2);
  a2=arg[2]? arg[2][7] : 0;
  a2=(a2*a4);
  a0=(a0+a2);
  a2=arg[2]? arg[2][10] : 0;
  a2=(a2*a5);
  a6=arg[2]? arg[2][13] : 0;
  a6=(a6*a7);
  a2=(a2+a6);
  a6=arg[2]? arg[2][16] : 0;
  a6=(a6*a8);
  a2=(a2+a6);
  a6=arg[2]? arg[2][19] : 0;
  a6=(a6*a9);
  a2=(a2+a6);
  a6=arg[2]? arg[2][22] : 0;
  a6=(a6*a10);
  a2=(a2+a6);
  a6=arg[2]? arg[2][25] : 0;
  a6=(a6*a11);
  a2=(a2+a6);
  a6=arg[2]? arg[2][28] : 0;
  a6=(a6*a12);
  a2=(a2+a6);
  a6=arg[2]? arg[2][31] : 0;
  a6=(a6*a13);
  a2=(a2+a6);
  a6=arg[2]? arg[2][34] : 0;
  a6=(a6*a14);
  a2=(a2+a6);
  a6=arg[2]? arg[2][37] : 0;
  a6=(a6*a15);
  a2=(a2+a6);
  a6=arg[2]? arg[2][40] : 0;
  a6=(a6*a16);
  a2=(a2+a6);
  a0=(a0+a2);
  a2=arg[2]? arg[2][43] : 0;
  a0=(a0+a2);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[2]? arg[2][2] : 0;
  a0=(a0*a1);
  a1=arg[2]? arg[2][5] : 0;
  a1=(a1*a3);
  a0=(a0+a1);
  a1=arg[2]? arg[2][8] : 0;
  a1=(a1*a4);
  a0=(a0+a1);
  a1=arg[2]? arg[2][11] : 0;
  a1=(a1*a5);
  a5=arg[2]? arg[2][14] : 0;
  a5=(a5*a7);
  a1=(a1+a5);
  a5=arg[2]? arg[2][17] : 0;
  a5=(a5*a8);
  a1=(a1+a5);
  a5=arg[2]? arg[2][20] : 0;
  a5=(a5*a9);
  a1=(a1+a5);
  a5=arg[2]? arg[2][23] : 0;
  a5=(a5*a10);
  a1=(a1+a5);
  a5=arg[2]? arg[2][26] : 0;
  a5=(a5*a11);
  a1=(a1+a5);
  a5=arg[2]? arg[2][29] : 0;
  a5=(a5*a12);
  a1=(a1+a5);
  a5=arg[2]? arg[2][32] : 0;
  a5=(a5*a13);
  a1=(a1+a5);
  a5=arg[2]? arg[2][35] : 0;
  a5=(a5*a14);
  a1=(a1+a5);
  a5=arg[2]? arg[2][38] : 0;
  a5=(a5*a15);
  a1=(a1+a5);
  a5=arg[2]? arg[2][41] : 0;
  a5=(a5*a16);
  a1=(a1+a5);
  a0=(a0+a1);
  a1=arg[2]? arg[2][44] : 0;
  a0=(a0+a1);
  if (res[0]!=0) res[0][2]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int msds_RAMPC_v1_dyn_disc_phi_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int msds_RAMPC_v1_dyn_disc_phi_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int msds_RAMPC_v1_dyn_disc_phi_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void msds_RAMPC_v1_dyn_disc_phi_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int msds_RAMPC_v1_dyn_disc_phi_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void msds_RAMPC_v1_dyn_disc_phi_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void msds_RAMPC_v1_dyn_disc_phi_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void msds_RAMPC_v1_dyn_disc_phi_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int msds_RAMPC_v1_dyn_disc_phi_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int msds_RAMPC_v1_dyn_disc_phi_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real msds_RAMPC_v1_dyn_disc_phi_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* msds_RAMPC_v1_dyn_disc_phi_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* msds_RAMPC_v1_dyn_disc_phi_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* msds_RAMPC_v1_dyn_disc_phi_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* msds_RAMPC_v1_dyn_disc_phi_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int msds_RAMPC_v1_dyn_disc_phi_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif