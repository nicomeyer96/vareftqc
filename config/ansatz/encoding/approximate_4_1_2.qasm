//d
//a0,a1,a2
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit a2;
h a2;
cx d,a0;
cx a2,a1;
cx a2,a0;
cx a2,d;
