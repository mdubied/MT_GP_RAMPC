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
  #define CASADI_PREFIX(ID) quadrotor_RAMPC_v1_dyn_disc_phi_fun_ ## ID
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

static const casadi_int casadi_s0[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s1[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s2[136] = {132, 1, 0, 132, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131};

/* quadrotor_RAMPC_v1_dyn_disc_phi_fun:(i0[7],i1[10],i2[132])->(o0[7]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[2]? arg[2][0] : 0;
  a1=arg[0]? arg[0][0] : 0;
  a0=(a0*a1);
  a2=arg[2]? arg[2][7] : 0;
  a3=arg[0]? arg[0][1] : 0;
  a2=(a2*a3);
  a0=(a0+a2);
  a2=arg[2]? arg[2][14] : 0;
  a4=arg[0]? arg[0][2] : 0;
  a2=(a2*a4);
  a0=(a0+a2);
  a2=arg[2]? arg[2][21] : 0;
  a5=arg[0]? arg[0][3] : 0;
  a2=(a2*a5);
  a0=(a0+a2);
  a2=arg[2]? arg[2][28] : 0;
  a6=arg[0]? arg[0][4] : 0;
  a2=(a2*a6);
  a0=(a0+a2);
  a2=arg[2]? arg[2][35] : 0;
  a7=arg[0]? arg[0][5] : 0;
  a2=(a2*a7);
  a0=(a0+a2);
  a2=arg[2]? arg[2][42] : 0;
  a8=arg[0]? arg[0][6] : 0;
  a2=(a2*a8);
  a0=(a0+a2);
  a2=arg[2]? arg[2][49] : 0;
  a9=arg[1]? arg[1][0] : 0;
  a2=(a2*a9);
  a10=arg[2]? arg[2][56] : 0;
  a11=arg[1]? arg[1][1] : 0;
  a10=(a10*a11);
  a2=(a2+a10);
  a10=arg[2]? arg[2][63] : 0;
  a12=arg[1]? arg[1][2] : 0;
  a10=(a10*a12);
  a2=(a2+a10);
  a10=arg[2]? arg[2][70] : 0;
  a13=arg[1]? arg[1][3] : 0;
  a10=(a10*a13);
  a2=(a2+a10);
  a10=arg[2]? arg[2][77] : 0;
  a14=arg[1]? arg[1][4] : 0;
  a10=(a10*a14);
  a2=(a2+a10);
  a10=arg[2]? arg[2][84] : 0;
  a15=arg[1]? arg[1][5] : 0;
  a10=(a10*a15);
  a2=(a2+a10);
  a10=arg[2]? arg[2][91] : 0;
  a16=arg[1]? arg[1][6] : 0;
  a10=(a10*a16);
  a2=(a2+a10);
  a10=arg[2]? arg[2][98] : 0;
  a17=arg[1]? arg[1][7] : 0;
  a10=(a10*a17);
  a2=(a2+a10);
  a10=arg[2]? arg[2][105] : 0;
  a18=arg[1]? arg[1][8] : 0;
  a10=(a10*a18);
  a2=(a2+a10);
  a10=arg[2]? arg[2][112] : 0;
  a19=arg[1]? arg[1][9] : 0;
  a10=(a10*a19);
  a2=(a2+a10);
  a0=(a0+a2);
  a2=arg[2]? arg[2][119] : 0;
  a0=(a0+a2);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[2]? arg[2][1] : 0;
  a0=(a0*a1);
  a2=arg[2]? arg[2][8] : 0;
  a2=(a2*a3);
  a0=(a0+a2);
  a2=arg[2]? arg[2][15] : 0;
  a2=(a2*a4);
  a0=(a0+a2);
  a2=arg[2]? arg[2][22] : 0;
  a2=(a2*a5);
  a0=(a0+a2);
  a2=arg[2]? arg[2][29] : 0;
  a2=(a2*a6);
  a0=(a0+a2);
  a2=arg[2]? arg[2][36] : 0;
  a2=(a2*a7);
  a0=(a0+a2);
  a2=arg[2]? arg[2][43] : 0;
  a2=(a2*a8);
  a0=(a0+a2);
  a2=arg[2]? arg[2][50] : 0;
  a2=(a2*a9);
  a10=arg[2]? arg[2][57] : 0;
  a10=(a10*a11);
  a2=(a2+a10);
  a10=arg[2]? arg[2][64] : 0;
  a10=(a10*a12);
  a2=(a2+a10);
  a10=arg[2]? arg[2][71] : 0;
  a10=(a10*a13);
  a2=(a2+a10);
  a10=arg[2]? arg[2][78] : 0;
  a10=(a10*a14);
  a2=(a2+a10);
  a10=arg[2]? arg[2][85] : 0;
  a10=(a10*a15);
  a2=(a2+a10);
  a10=arg[2]? arg[2][92] : 0;
  a10=(a10*a16);
  a2=(a2+a10);
  a10=arg[2]? arg[2][99] : 0;
  a10=(a10*a17);
  a2=(a2+a10);
  a10=arg[2]? arg[2][106] : 0;
  a10=(a10*a18);
  a2=(a2+a10);
  a10=arg[2]? arg[2][113] : 0;
  a10=(a10*a19);
  a2=(a2+a10);
  a0=(a0+a2);
  a2=arg[2]? arg[2][120] : 0;
  a0=(a0+a2);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[2]? arg[2][2] : 0;
  a0=(a0*a1);
  a2=arg[2]? arg[2][9] : 0;
  a2=(a2*a3);
  a0=(a0+a2);
  a2=arg[2]? arg[2][16] : 0;
  a2=(a2*a4);
  a0=(a0+a2);
  a2=arg[2]? arg[2][23] : 0;
  a2=(a2*a5);
  a0=(a0+a2);
  a2=arg[2]? arg[2][30] : 0;
  a2=(a2*a6);
  a0=(a0+a2);
  a2=arg[2]? arg[2][37] : 0;
  a2=(a2*a7);
  a0=(a0+a2);
  a2=arg[2]? arg[2][44] : 0;
  a2=(a2*a8);
  a0=(a0+a2);
  a2=arg[2]? arg[2][51] : 0;
  a2=(a2*a9);
  a10=arg[2]? arg[2][58] : 0;
  a10=(a10*a11);
  a2=(a2+a10);
  a10=arg[2]? arg[2][65] : 0;
  a10=(a10*a12);
  a2=(a2+a10);
  a10=arg[2]? arg[2][72] : 0;
  a10=(a10*a13);
  a2=(a2+a10);
  a10=arg[2]? arg[2][79] : 0;
  a10=(a10*a14);
  a2=(a2+a10);
  a10=arg[2]? arg[2][86] : 0;
  a10=(a10*a15);
  a2=(a2+a10);
  a10=arg[2]? arg[2][93] : 0;
  a10=(a10*a16);
  a2=(a2+a10);
  a10=arg[2]? arg[2][100] : 0;
  a10=(a10*a17);
  a2=(a2+a10);
  a10=arg[2]? arg[2][107] : 0;
  a10=(a10*a18);
  a2=(a2+a10);
  a10=arg[2]? arg[2][114] : 0;
  a10=(a10*a19);
  a2=(a2+a10);
  a0=(a0+a2);
  a2=arg[2]? arg[2][121] : 0;
  a0=(a0+a2);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[2]? arg[2][3] : 0;
  a0=(a0*a1);
  a2=arg[2]? arg[2][10] : 0;
  a2=(a2*a3);
  a0=(a0+a2);
  a2=arg[2]? arg[2][17] : 0;
  a2=(a2*a4);
  a0=(a0+a2);
  a2=arg[2]? arg[2][24] : 0;
  a2=(a2*a5);
  a0=(a0+a2);
  a2=arg[2]? arg[2][31] : 0;
  a2=(a2*a6);
  a0=(a0+a2);
  a2=arg[2]? arg[2][38] : 0;
  a2=(a2*a7);
  a0=(a0+a2);
  a2=arg[2]? arg[2][45] : 0;
  a2=(a2*a8);
  a0=(a0+a2);
  a2=arg[2]? arg[2][52] : 0;
  a2=(a2*a9);
  a10=arg[2]? arg[2][59] : 0;
  a10=(a10*a11);
  a2=(a2+a10);
  a10=arg[2]? arg[2][66] : 0;
  a10=(a10*a12);
  a2=(a2+a10);
  a10=arg[2]? arg[2][73] : 0;
  a10=(a10*a13);
  a2=(a2+a10);
  a10=arg[2]? arg[2][80] : 0;
  a10=(a10*a14);
  a2=(a2+a10);
  a10=arg[2]? arg[2][87] : 0;
  a10=(a10*a15);
  a2=(a2+a10);
  a10=arg[2]? arg[2][94] : 0;
  a10=(a10*a16);
  a2=(a2+a10);
  a10=arg[2]? arg[2][101] : 0;
  a10=(a10*a17);
  a2=(a2+a10);
  a10=arg[2]? arg[2][108] : 0;
  a10=(a10*a18);
  a2=(a2+a10);
  a10=arg[2]? arg[2][115] : 0;
  a10=(a10*a19);
  a2=(a2+a10);
  a0=(a0+a2);
  a2=arg[2]? arg[2][122] : 0;
  a0=(a0+a2);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[2]? arg[2][4] : 0;
  a0=(a0*a1);
  a2=arg[2]? arg[2][11] : 0;
  a2=(a2*a3);
  a0=(a0+a2);
  a2=arg[2]? arg[2][18] : 0;
  a2=(a2*a4);
  a0=(a0+a2);
  a2=arg[2]? arg[2][25] : 0;
  a2=(a2*a5);
  a0=(a0+a2);
  a2=arg[2]? arg[2][32] : 0;
  a2=(a2*a6);
  a0=(a0+a2);
  a2=arg[2]? arg[2][39] : 0;
  a2=(a2*a7);
  a0=(a0+a2);
  a2=arg[2]? arg[2][46] : 0;
  a2=(a2*a8);
  a0=(a0+a2);
  a2=arg[2]? arg[2][53] : 0;
  a2=(a2*a9);
  a10=arg[2]? arg[2][60] : 0;
  a10=(a10*a11);
  a2=(a2+a10);
  a10=arg[2]? arg[2][67] : 0;
  a10=(a10*a12);
  a2=(a2+a10);
  a10=arg[2]? arg[2][74] : 0;
  a10=(a10*a13);
  a2=(a2+a10);
  a10=arg[2]? arg[2][81] : 0;
  a10=(a10*a14);
  a2=(a2+a10);
  a10=arg[2]? arg[2][88] : 0;
  a10=(a10*a15);
  a2=(a2+a10);
  a10=arg[2]? arg[2][95] : 0;
  a10=(a10*a16);
  a2=(a2+a10);
  a10=arg[2]? arg[2][102] : 0;
  a10=(a10*a17);
  a2=(a2+a10);
  a10=arg[2]? arg[2][109] : 0;
  a10=(a10*a18);
  a2=(a2+a10);
  a10=arg[2]? arg[2][116] : 0;
  a10=(a10*a19);
  a2=(a2+a10);
  a0=(a0+a2);
  a2=arg[2]? arg[2][123] : 0;
  a0=(a0+a2);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[2]? arg[2][5] : 0;
  a0=(a0*a1);
  a2=arg[2]? arg[2][12] : 0;
  a2=(a2*a3);
  a0=(a0+a2);
  a2=arg[2]? arg[2][19] : 0;
  a2=(a2*a4);
  a0=(a0+a2);
  a2=arg[2]? arg[2][26] : 0;
  a2=(a2*a5);
  a0=(a0+a2);
  a2=arg[2]? arg[2][33] : 0;
  a2=(a2*a6);
  a0=(a0+a2);
  a2=arg[2]? arg[2][40] : 0;
  a2=(a2*a7);
  a0=(a0+a2);
  a2=arg[2]? arg[2][47] : 0;
  a2=(a2*a8);
  a0=(a0+a2);
  a2=arg[2]? arg[2][54] : 0;
  a2=(a2*a9);
  a10=arg[2]? arg[2][61] : 0;
  a10=(a10*a11);
  a2=(a2+a10);
  a10=arg[2]? arg[2][68] : 0;
  a10=(a10*a12);
  a2=(a2+a10);
  a10=arg[2]? arg[2][75] : 0;
  a10=(a10*a13);
  a2=(a2+a10);
  a10=arg[2]? arg[2][82] : 0;
  a10=(a10*a14);
  a2=(a2+a10);
  a10=arg[2]? arg[2][89] : 0;
  a10=(a10*a15);
  a2=(a2+a10);
  a10=arg[2]? arg[2][96] : 0;
  a10=(a10*a16);
  a2=(a2+a10);
  a10=arg[2]? arg[2][103] : 0;
  a10=(a10*a17);
  a2=(a2+a10);
  a10=arg[2]? arg[2][110] : 0;
  a10=(a10*a18);
  a2=(a2+a10);
  a10=arg[2]? arg[2][117] : 0;
  a10=(a10*a19);
  a2=(a2+a10);
  a0=(a0+a2);
  a2=arg[2]? arg[2][124] : 0;
  a0=(a0+a2);
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[2]? arg[2][6] : 0;
  a0=(a0*a1);
  a1=arg[2]? arg[2][13] : 0;
  a1=(a1*a3);
  a0=(a0+a1);
  a1=arg[2]? arg[2][20] : 0;
  a1=(a1*a4);
  a0=(a0+a1);
  a1=arg[2]? arg[2][27] : 0;
  a1=(a1*a5);
  a0=(a0+a1);
  a1=arg[2]? arg[2][34] : 0;
  a1=(a1*a6);
  a0=(a0+a1);
  a1=arg[2]? arg[2][41] : 0;
  a1=(a1*a7);
  a0=(a0+a1);
  a1=arg[2]? arg[2][48] : 0;
  a1=(a1*a8);
  a0=(a0+a1);
  a1=arg[2]? arg[2][55] : 0;
  a1=(a1*a9);
  a9=arg[2]? arg[2][62] : 0;
  a9=(a9*a11);
  a1=(a1+a9);
  a9=arg[2]? arg[2][69] : 0;
  a9=(a9*a12);
  a1=(a1+a9);
  a9=arg[2]? arg[2][76] : 0;
  a9=(a9*a13);
  a1=(a1+a9);
  a9=arg[2]? arg[2][83] : 0;
  a9=(a9*a14);
  a1=(a1+a9);
  a9=arg[2]? arg[2][90] : 0;
  a9=(a9*a15);
  a1=(a1+a9);
  a9=arg[2]? arg[2][97] : 0;
  a9=(a9*a16);
  a1=(a1+a9);
  a9=arg[2]? arg[2][104] : 0;
  a9=(a9*a17);
  a1=(a1+a9);
  a9=arg[2]? arg[2][111] : 0;
  a9=(a9*a18);
  a1=(a1+a9);
  a9=arg[2]? arg[2][118] : 0;
  a9=(a9*a19);
  a1=(a1+a9);
  a0=(a0+a1);
  a1=arg[2]? arg[2][125] : 0;
  a0=(a0+a1);
  if (res[0]!=0) res[0][6]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v1_dyn_disc_phi_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v1_dyn_disc_phi_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v1_dyn_disc_phi_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v1_dyn_disc_phi_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_RAMPC_v1_dyn_disc_phi_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_RAMPC_v1_dyn_disc_phi_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real quadrotor_RAMPC_v1_dyn_disc_phi_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_RAMPC_v1_dyn_disc_phi_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_RAMPC_v1_dyn_disc_phi_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_RAMPC_v1_dyn_disc_phi_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_RAMPC_v1_dyn_disc_phi_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
