//d
//a0,a1,a2,a3,a4,a5
//d_t
//a0_t,a1_t,a2_t,a3_t,a4_t,a5_t
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit a2;
qubit a3;
qubit a4;
qubit a5;
qubit d_t;
qubit a0_t;
qubit a1_t;
qubit a2_t;
qubit a3_t;
qubit a4_t;
qubit a5_t;
cz d,d_t;
cz a0,a0_t;
cz a1,a1_t;
cz a2,a2_t;
cz a3,a3_t;
cz a4,a4_t;
cz a5,a5_t;
