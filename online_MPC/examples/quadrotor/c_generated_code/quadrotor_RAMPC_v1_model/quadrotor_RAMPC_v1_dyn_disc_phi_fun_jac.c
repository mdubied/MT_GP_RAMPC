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
  #define CASADI_PREFIX(ID) quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_ ## ID
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
#define casadi_s3 CASADI_PREFIX(s3)

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
static const casadi_int casadi_s3[129] = {17, 7, 0, 17, 34, 51, 68, 85, 102, 119, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

/* quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac:(i0[7],i1[10],i2[132])->(o0[7],o1[17x7]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[2]? arg[2][0] : 0;
  a1=arg[0]? arg[0][0] : 0;
  a2=(a0*a1);
  a3=arg[2]? arg[2][7] : 0;
  a4=arg[0]? arg[0][1] : 0;
  a5=(a3*a4);
  a2=(a2+a5);
  a5=arg[2]? arg[2][14] : 0;
  a6=arg[0]? arg[0][2] : 0;
  a7=(a5*a6);
  a2=(a2+a7);
  a7=arg[2]? arg[2][21] : 0;
  a8=arg[0]? arg[0][3] : 0;
  a9=(a7*a8);
  a2=(a2+a9);
  a9=arg[2]? arg[2][28] : 0;
  a10=arg[0]? arg[0][4] : 0;
  a11=(a9*a10);
  a2=(a2+a11);
  a11=arg[2]? arg[2][35] : 0;
  a12=arg[0]? arg[0][5] : 0;
  a13=(a11*a12);
  a2=(a2+a13);
  a13=arg[2]? arg[2][42] : 0;
  a14=arg[0]? arg[0][6] : 0;
  a15=(a13*a14);
  a2=(a2+a15);
  a15=arg[2]? arg[2][49] : 0;
  a16=arg[1]? arg[1][0] : 0;
  a17=(a15*a16);
  a18=arg[2]? arg[2][56] : 0;
  a19=arg[1]? arg[1][1] : 0;
  a20=(a18*a19);
  a17=(a17+a20);
  a20=arg[2]? arg[2][63] : 0;
  a21=arg[1]? arg[1][2] : 0;
  a22=(a20*a21);
  a17=(a17+a22);
  a22=arg[2]? arg[2][70] : 0;
  a23=arg[1]? arg[1][3] : 0;
  a24=(a22*a23);
  a17=(a17+a24);
  a24=arg[2]? arg[2][77] : 0;
  a25=arg[1]? arg[1][4] : 0;
  a26=(a24*a25);
  a17=(a17+a26);
  a26=arg[2]? arg[2][84] : 0;
  a27=arg[1]? arg[1][5] : 0;
  a28=(a26*a27);
  a17=(a17+a28);
  a28=arg[2]? arg[2][91] : 0;
  a29=arg[1]? arg[1][6] : 0;
  a30=(a28*a29);
  a17=(a17+a30);
  a30=arg[2]? arg[2][98] : 0;
  a31=arg[1]? arg[1][7] : 0;
  a32=(a30*a31);
  a17=(a17+a32);
  a32=arg[2]? arg[2][105] : 0;
  a33=arg[1]? arg[1][8] : 0;
  a34=(a32*a33);
  a17=(a17+a34);
  a34=arg[2]? arg[2][112] : 0;
  a35=arg[1]? arg[1][9] : 0;
  a36=(a34*a35);
  a17=(a17+a36);
  a2=(a2+a17);
  a17=arg[2]? arg[2][119] : 0;
  a2=(a2+a17);
  if (res[0]!=0) res[0][0]=a2;
  a2=arg[2]? arg[2][1] : 0;
  a17=(a2*a1);
  a36=arg[2]? arg[2][8] : 0;
  a37=(a36*a4);
  a17=(a17+a37);
  a37=arg[2]? arg[2][15] : 0;
  a38=(a37*a6);
  a17=(a17+a38);
  a38=arg[2]? arg[2][22] : 0;
  a39=(a38*a8);
  a17=(a17+a39);
  a39=arg[2]? arg[2][29] : 0;
  a40=(a39*a10);
  a17=(a17+a40);
  a40=arg[2]? arg[2][36] : 0;
  a41=(a40*a12);
  a17=(a17+a41);
  a41=arg[2]? arg[2][43] : 0;
  a42=(a41*a14);
  a17=(a17+a42);
  a42=arg[2]? arg[2][50] : 0;
  a43=(a42*a16);
  a44=arg[2]? arg[2][57] : 0;
  a45=(a44*a19);
  a43=(a43+a45);
  a45=arg[2]? arg[2][64] : 0;
  a46=(a45*a21);
  a43=(a43+a46);
  a46=arg[2]? arg[2][71] : 0;
  a47=(a46*a23);
  a43=(a43+a47);
  a47=arg[2]? arg[2][78] : 0;
  a48=(a47*a25);
  a43=(a43+a48);
  a48=arg[2]? arg[2][85] : 0;
  a49=(a48*a27);
  a43=(a43+a49);
  a49=arg[2]? arg[2][92] : 0;
  a50=(a49*a29);
  a43=(a43+a50);
  a50=arg[2]? arg[2][99] : 0;
  a51=(a50*a31);
  a43=(a43+a51);
  a51=arg[2]? arg[2][106] : 0;
  a52=(a51*a33);
  a43=(a43+a52);
  a52=arg[2]? arg[2][113] : 0;
  a53=(a52*a35);
  a43=(a43+a53);
  a17=(a17+a43);
  a43=arg[2]? arg[2][120] : 0;
  a17=(a17+a43);
  if (res[0]!=0) res[0][1]=a17;
  a17=arg[2]? arg[2][2] : 0;
  a43=(a17*a1);
  a53=arg[2]? arg[2][9] : 0;
  a54=(a53*a4);
  a43=(a43+a54);
  a54=arg[2]? arg[2][16] : 0;
  a55=(a54*a6);
  a43=(a43+a55);
  a55=arg[2]? arg[2][23] : 0;
  a56=(a55*a8);
  a43=(a43+a56);
  a56=arg[2]? arg[2][30] : 0;
  a57=(a56*a10);
  a43=(a43+a57);
  a57=arg[2]? arg[2][37] : 0;
  a58=(a57*a12);
  a43=(a43+a58);
  a58=arg[2]? arg[2][44] : 0;
  a59=(a58*a14);
  a43=(a43+a59);
  a59=arg[2]? arg[2][51] : 0;
  a60=(a59*a16);
  a61=arg[2]? arg[2][58] : 0;
  a62=(a61*a19);
  a60=(a60+a62);
  a62=arg[2]? arg[2][65] : 0;
  a63=(a62*a21);
  a60=(a60+a63);
  a63=arg[2]? arg[2][72] : 0;
  a64=(a63*a23);
  a60=(a60+a64);
  a64=arg[2]? arg[2][79] : 0;
  a65=(a64*a25);
  a60=(a60+a65);
  a65=arg[2]? arg[2][86] : 0;
  a66=(a65*a27);
  a60=(a60+a66);
  a66=arg[2]? arg[2][93] : 0;
  a67=(a66*a29);
  a60=(a60+a67);
  a67=arg[2]? arg[2][100] : 0;
  a68=(a67*a31);
  a60=(a60+a68);
  a68=arg[2]? arg[2][107] : 0;
  a69=(a68*a33);
  a60=(a60+a69);
  a69=arg[2]? arg[2][114] : 0;
  a70=(a69*a35);
  a60=(a60+a70);
  a43=(a43+a60);
  a60=arg[2]? arg[2][121] : 0;
  a43=(a43+a60);
  if (res[0]!=0) res[0][2]=a43;
  a43=arg[2]? arg[2][3] : 0;
  a60=(a43*a1);
  a70=arg[2]? arg[2][10] : 0;
  a71=(a70*a4);
  a60=(a60+a71);
  a71=arg[2]? arg[2][17] : 0;
  a72=(a71*a6);
  a60=(a60+a72);
  a72=arg[2]? arg[2][24] : 0;
  a73=(a72*a8);
  a60=(a60+a73);
  a73=arg[2]? arg[2][31] : 0;
  a74=(a73*a10);
  a60=(a60+a74);
  a74=arg[2]? arg[2][38] : 0;
  a75=(a74*a12);
  a60=(a60+a75);
  a75=arg[2]? arg[2][45] : 0;
  a76=(a75*a14);
  a60=(a60+a76);
  a76=arg[2]? arg[2][52] : 0;
  a77=(a76*a16);
  a78=arg[2]? arg[2][59] : 0;
  a79=(a78*a19);
  a77=(a77+a79);
  a79=arg[2]? arg[2][66] : 0;
  a80=(a79*a21);
  a77=(a77+a80);
  a80=arg[2]? arg[2][73] : 0;
  a81=(a80*a23);
  a77=(a77+a81);
  a81=arg[2]? arg[2][80] : 0;
  a82=(a81*a25);
  a77=(a77+a82);
  a82=arg[2]? arg[2][87] : 0;
  a83=(a82*a27);
  a77=(a77+a83);
  a83=arg[2]? arg[2][94] : 0;
  a84=(a83*a29);
  a77=(a77+a84);
  a84=arg[2]? arg[2][101] : 0;
  a85=(a84*a31);
  a77=(a77+a85);
  a85=arg[2]? arg[2][108] : 0;
  a86=(a85*a33);
  a77=(a77+a86);
  a86=arg[2]? arg[2][115] : 0;
  a87=(a86*a35);
  a77=(a77+a87);
  a60=(a60+a77);
  a77=arg[2]? arg[2][122] : 0;
  a60=(a60+a77);
  if (res[0]!=0) res[0][3]=a60;
  a60=arg[2]? arg[2][4] : 0;
  a77=(a60*a1);
  a87=arg[2]? arg[2][11] : 0;
  a88=(a87*a4);
  a77=(a77+a88);
  a88=arg[2]? arg[2][18] : 0;
  a89=(a88*a6);
  a77=(a77+a89);
  a89=arg[2]? arg[2][25] : 0;
  a90=(a89*a8);
  a77=(a77+a90);
  a90=arg[2]? arg[2][32] : 0;
  a91=(a90*a10);
  a77=(a77+a91);
  a91=arg[2]? arg[2][39] : 0;
  a92=(a91*a12);
  a77=(a77+a92);
  a92=arg[2]? arg[2][46] : 0;
  a93=(a92*a14);
  a77=(a77+a93);
  a93=arg[2]? arg[2][53] : 0;
  a94=(a93*a16);
  a95=arg[2]? arg[2][60] : 0;
  a96=(a95*a19);
  a94=(a94+a96);
  a96=arg[2]? arg[2][67] : 0;
  a97=(a96*a21);
  a94=(a94+a97);
  a97=arg[2]? arg[2][74] : 0;
  a98=(a97*a23);
  a94=(a94+a98);
  a98=arg[2]? arg[2][81] : 0;
  a99=(a98*a25);
  a94=(a94+a99);
  a99=arg[2]? arg[2][88] : 0;
  a100=(a99*a27);
  a94=(a94+a100);
  a100=arg[2]? arg[2][95] : 0;
  a101=(a100*a29);
  a94=(a94+a101);
  a101=arg[2]? arg[2][102] : 0;
  a102=(a101*a31);
  a94=(a94+a102);
  a102=arg[2]? arg[2][109] : 0;
  a103=(a102*a33);
  a94=(a94+a103);
  a103=arg[2]? arg[2][116] : 0;
  a104=(a103*a35);
  a94=(a94+a104);
  a77=(a77+a94);
  a94=arg[2]? arg[2][123] : 0;
  a77=(a77+a94);
  if (res[0]!=0) res[0][4]=a77;
  a77=arg[2]? arg[2][5] : 0;
  a94=(a77*a1);
  a104=arg[2]? arg[2][12] : 0;
  a105=(a104*a4);
  a94=(a94+a105);
  a105=arg[2]? arg[2][19] : 0;
  a106=(a105*a6);
  a94=(a94+a106);
  a106=arg[2]? arg[2][26] : 0;
  a107=(a106*a8);
  a94=(a94+a107);
  a107=arg[2]? arg[2][33] : 0;
  a108=(a107*a10);
  a94=(a94+a108);
  a108=arg[2]? arg[2][40] : 0;
  a109=(a108*a12);
  a94=(a94+a109);
  a109=arg[2]? arg[2][47] : 0;
  a110=(a109*a14);
  a94=(a94+a110);
  a110=arg[2]? arg[2][54] : 0;
  a111=(a110*a16);
  a112=arg[2]? arg[2][61] : 0;
  a113=(a112*a19);
  a111=(a111+a113);
  a113=arg[2]? arg[2][68] : 0;
  a114=(a113*a21);
  a111=(a111+a114);
  a114=arg[2]? arg[2][75] : 0;
  a115=(a114*a23);
  a111=(a111+a115);
  a115=arg[2]? arg[2][82] : 0;
  a116=(a115*a25);
  a111=(a111+a116);
  a116=arg[2]? arg[2][89] : 0;
  a117=(a116*a27);
  a111=(a111+a117);
  a117=arg[2]? arg[2][96] : 0;
  a118=(a117*a29);
  a111=(a111+a118);
  a118=arg[2]? arg[2][103] : 0;
  a119=(a118*a31);
  a111=(a111+a119);
  a119=arg[2]? arg[2][110] : 0;
  a120=(a119*a33);
  a111=(a111+a120);
  a120=arg[2]? arg[2][117] : 0;
  a121=(a120*a35);
  a111=(a111+a121);
  a94=(a94+a111);
  a111=arg[2]? arg[2][124] : 0;
  a94=(a94+a111);
  if (res[0]!=0) res[0][5]=a94;
  a94=arg[2]? arg[2][6] : 0;
  a1=(a94*a1);
  a111=arg[2]? arg[2][13] : 0;
  a4=(a111*a4);
  a1=(a1+a4);
  a4=arg[2]? arg[2][20] : 0;
  a6=(a4*a6);
  a1=(a1+a6);
  a6=arg[2]? arg[2][27] : 0;
  a8=(a6*a8);
  a1=(a1+a8);
  a8=arg[2]? arg[2][34] : 0;
  a10=(a8*a10);
  a1=(a1+a10);
  a10=arg[2]? arg[2][41] : 0;
  a12=(a10*a12);
  a1=(a1+a12);
  a12=arg[2]? arg[2][48] : 0;
  a14=(a12*a14);
  a1=(a1+a14);
  a14=arg[2]? arg[2][55] : 0;
  a16=(a14*a16);
  a121=arg[2]? arg[2][62] : 0;
  a19=(a121*a19);
  a16=(a16+a19);
  a19=arg[2]? arg[2][69] : 0;
  a21=(a19*a21);
  a16=(a16+a21);
  a21=arg[2]? arg[2][76] : 0;
  a23=(a21*a23);
  a16=(a16+a23);
  a23=arg[2]? arg[2][83] : 0;
  a25=(a23*a25);
  a16=(a16+a25);
  a25=arg[2]? arg[2][90] : 0;
  a27=(a25*a27);
  a16=(a16+a27);
  a27=arg[2]? arg[2][97] : 0;
  a29=(a27*a29);
  a16=(a16+a29);
  a29=arg[2]? arg[2][104] : 0;
  a31=(a29*a31);
  a16=(a16+a31);
  a31=arg[2]? arg[2][111] : 0;
  a33=(a31*a33);
  a16=(a16+a33);
  a33=arg[2]? arg[2][118] : 0;
  a35=(a33*a35);
  a16=(a16+a35);
  a1=(a1+a16);
  a16=arg[2]? arg[2][125] : 0;
  a1=(a1+a16);
  if (res[0]!=0) res[0][6]=a1;
  if (res[1]!=0) res[1][0]=a15;
  if (res[1]!=0) res[1][1]=a18;
  if (res[1]!=0) res[1][2]=a20;
  if (res[1]!=0) res[1][3]=a22;
  if (res[1]!=0) res[1][4]=a24;
  if (res[1]!=0) res[1][5]=a26;
  if (res[1]!=0) res[1][6]=a28;
  if (res[1]!=0) res[1][7]=a30;
  if (res[1]!=0) res[1][8]=a32;
  if (res[1]!=0) res[1][9]=a34;
  if (res[1]!=0) res[1][10]=a0;
  if (res[1]!=0) res[1][11]=a3;
  if (res[1]!=0) res[1][12]=a5;
  if (res[1]!=0) res[1][13]=a7;
  if (res[1]!=0) res[1][14]=a9;
  if (res[1]!=0) res[1][15]=a11;
  if (res[1]!=0) res[1][16]=a13;
  if (res[1]!=0) res[1][17]=a42;
  if (res[1]!=0) res[1][18]=a44;
  if (res[1]!=0) res[1][19]=a45;
  if (res[1]!=0) res[1][20]=a46;
  if (res[1]!=0) res[1][21]=a47;
  if (res[1]!=0) res[1][22]=a48;
  if (res[1]!=0) res[1][23]=a49;
  if (res[1]!=0) res[1][24]=a50;
  if (res[1]!=0) res[1][25]=a51;
  if (res[1]!=0) res[1][26]=a52;
  if (res[1]!=0) res[1][27]=a2;
  if (res[1]!=0) res[1][28]=a36;
  if (res[1]!=0) res[1][29]=a37;
  if (res[1]!=0) res[1][30]=a38;
  if (res[1]!=0) res[1][31]=a39;
  if (res[1]!=0) res[1][32]=a40;
  if (res[1]!=0) res[1][33]=a41;
  if (res[1]!=0) res[1][34]=a59;
  if (res[1]!=0) res[1][35]=a61;
  if (res[1]!=0) res[1][36]=a62;
  if (res[1]!=0) res[1][37]=a63;
  if (res[1]!=0) res[1][38]=a64;
  if (res[1]!=0) res[1][39]=a65;
  if (res[1]!=0) res[1][40]=a66;
  if (res[1]!=0) res[1][41]=a67;
  if (res[1]!=0) res[1][42]=a68;
  if (res[1]!=0) res[1][43]=a69;
  if (res[1]!=0) res[1][44]=a17;
  if (res[1]!=0) res[1][45]=a53;
  if (res[1]!=0) res[1][46]=a54;
  if (res[1]!=0) res[1][47]=a55;
  if (res[1]!=0) res[1][48]=a56;
  if (res[1]!=0) res[1][49]=a57;
  if (res[1]!=0) res[1][50]=a58;
  if (res[1]!=0) res[1][51]=a76;
  if (res[1]!=0) res[1][52]=a78;
  if (res[1]!=0) res[1][53]=a79;
  if (res[1]!=0) res[1][54]=a80;
  if (res[1]!=0) res[1][55]=a81;
  if (res[1]!=0) res[1][56]=a82;
  if (res[1]!=0) res[1][57]=a83;
  if (res[1]!=0) res[1][58]=a84;
  if (res[1]!=0) res[1][59]=a85;
  if (res[1]!=0) res[1][60]=a86;
  if (res[1]!=0) res[1][61]=a43;
  if (res[1]!=0) res[1][62]=a70;
  if (res[1]!=0) res[1][63]=a71;
  if (res[1]!=0) res[1][64]=a72;
  if (res[1]!=0) res[1][65]=a73;
  if (res[1]!=0) res[1][66]=a74;
  if (res[1]!=0) res[1][67]=a75;
  if (res[1]!=0) res[1][68]=a93;
  if (res[1]!=0) res[1][69]=a95;
  if (res[1]!=0) res[1][70]=a96;
  if (res[1]!=0) res[1][71]=a97;
  if (res[1]!=0) res[1][72]=a98;
  if (res[1]!=0) res[1][73]=a99;
  if (res[1]!=0) res[1][74]=a100;
  if (res[1]!=0) res[1][75]=a101;
  if (res[1]!=0) res[1][76]=a102;
  if (res[1]!=0) res[1][77]=a103;
  if (res[1]!=0) res[1][78]=a60;
  if (res[1]!=0) res[1][79]=a87;
  if (res[1]!=0) res[1][80]=a88;
  if (res[1]!=0) res[1][81]=a89;
  if (res[1]!=0) res[1][82]=a90;
  if (res[1]!=0) res[1][83]=a91;
  if (res[1]!=0) res[1][84]=a92;
  if (res[1]!=0) res[1][85]=a110;
  if (res[1]!=0) res[1][86]=a112;
  if (res[1]!=0) res[1][87]=a113;
  if (res[1]!=0) res[1][88]=a114;
  if (res[1]!=0) res[1][89]=a115;
  if (res[1]!=0) res[1][90]=a116;
  if (res[1]!=0) res[1][91]=a117;
  if (res[1]!=0) res[1][92]=a118;
  if (res[1]!=0) res[1][93]=a119;
  if (res[1]!=0) res[1][94]=a120;
  if (res[1]!=0) res[1][95]=a77;
  if (res[1]!=0) res[1][96]=a104;
  if (res[1]!=0) res[1][97]=a105;
  if (res[1]!=0) res[1][98]=a106;
  if (res[1]!=0) res[1][99]=a107;
  if (res[1]!=0) res[1][100]=a108;
  if (res[1]!=0) res[1][101]=a109;
  if (res[1]!=0) res[1][102]=a14;
  if (res[1]!=0) res[1][103]=a121;
  if (res[1]!=0) res[1][104]=a19;
  if (res[1]!=0) res[1][105]=a21;
  if (res[1]!=0) res[1][106]=a23;
  if (res[1]!=0) res[1][107]=a25;
  if (res[1]!=0) res[1][108]=a27;
  if (res[1]!=0) res[1][109]=a29;
  if (res[1]!=0) res[1][110]=a31;
  if (res[1]!=0) res[1][111]=a33;
  if (res[1]!=0) res[1][112]=a94;
  if (res[1]!=0) res[1][113]=a111;
  if (res[1]!=0) res[1][114]=a4;
  if (res[1]!=0) res[1][115]=a6;
  if (res[1]!=0) res[1][116]=a8;
  if (res[1]!=0) res[1][117]=a10;
  if (res[1]!=0) res[1][118]=a12;
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
