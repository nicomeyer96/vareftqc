//d
//a0,a1,a2,a3
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit a2;
qubit a3;
z d;
h a1;
h a2;
sdg d;
cx a1,a3;
cx a2,a0;
h a0;
sdg a1;
cx a2,a3;
cx a0,d;
z a1;
s a2;
sdg a3;
s d;
s a0;
cx a3,d;
h a3;
cx a3,a0;
