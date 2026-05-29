//d
//a0,a1
//d_t
//a0_t,a1_t
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit d_t;
qubit a0_t;
qubit a1_t;
cx d,d_t;
cx a0,a0_t;
cx a1,a1_t;
