//d
//a0,a1,a2,a3,a4,a5,a6,a7
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit a2;
qubit a3;
qubit a4;
qubit a5;
qubit a6;
qubit a7;
cx d,a2;
cx d,a5;
h d;
h a2;
h a5;
cx d,a0;
cx d,a1;
cx a2,a3;
cx a2,a4;
cx a5,a6;
cx a5,a7;
