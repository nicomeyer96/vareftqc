//d
//a0,a1,a2,a3,a4,a5
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit a2;
qubit a3;
qubit a4;
qubit a5;
h a0;
h a1;
h a2;
cx d,a3;
cx d,a4;
cx a2,d;
cx a2,a3;
cx a2,a5;
cx a1,d;
cx a1,a4;
cx a1,a5;
cx a0,a3;
cx a0,a4;
cx a0,a5;
