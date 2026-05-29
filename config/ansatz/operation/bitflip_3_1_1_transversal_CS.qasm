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
ctrl @ sdg d_t,d;
ctrl @ sdg a0_t,a0;
ctrl @ sdg a1_t,a1;
