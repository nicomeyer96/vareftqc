//d
//a0,a1,a2
//d_t
//a0_t,a1_t,a2_t
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit a2;
qubit d_t;
qubit a0_t;
qubit a1_t;
qubit a2_t;
cx d,d_t;
cx a0,a0_t;
cx a1,a1_t;
cx a2,a2_t;
