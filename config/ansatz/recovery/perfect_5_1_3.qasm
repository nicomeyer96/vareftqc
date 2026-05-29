//d
//a0,a1,a2,a3
//r0,r1,r2,r3
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit a2;
qubit a3;
qubit r0;
qubit r1;
qubit r2;
qubit r3;
// SYNDROME EXTRACTION
// initial hadamard layer
h r0;
h r1;
h r2;
h r3;
// first syndrome wire (stabilizer XZZXI)
cx r0,d;
cz r0,a0;
cz r0,a1;
cx r0,a2;
// second syndrome wire (stabilizer IXZZX)
cx r1,a0;
cz r1,a1;
cz r1,a2;
cx r1,a3;
// third syndrome wire (stabilizer XIXZZ)
cx r2,d;
cx r2,a1;
cz r2,a2;
cz r2,a3;
// fourth syndrome wire (stabilizer ZXIXZ)
cz r3,d;
cx r3,a0;
cx r3,a2;
cz r3,a3;
// final Hadamard layer
h r0;
h r1;
h r2;
h r3;
// RECOVERY (COHERENT)
// correct for X errors
negctrl @ negctrl @ negctrl @ ctrl @ x r0, r1, r2, r3, d;
ctrl @ negctrl @ negctrl @ negctrl @ x r0, r1, r2, r3, a0;
ctrl @ ctrl @ negctrl @ negctrl @ x r0, r1, r2, r3, a1;
negctrl @ ctrl @ ctrl @ negctrl @ x r0, r1, r2, r3, a2;
negctrl @ negctrl @ ctrl @ ctrl @ x r0, r1, r2, r3, a3;
// correct for Z errors
ctrl @ negctrl @ ctrl @ negctrl @ z r0, r1, r2, r3, d;
negctrl @ ctrl @ negctrl @ ctrl @ z r0, r1, r2, r3, a0;
negctrl @ negctrl @ ctrl @ negctrl @ z r0, r1, r2, r3, a1;
ctrl @ negctrl @ negctrl @ ctrl @ z r0, r1, r2, r3, a2;
negctrl @ ctrl @ negctrl @ negctrl @ z r0, r1, r2, r3, a3;
// correct for Y errors
ctrl @ negctrl @ ctrl @ ctrl @ y r0, r1, r2, r3, d;
ctrl @ ctrl @ negctrl @ ctrl @ y r0, r1, r2, r3, a0;
ctrl @ ctrl @ ctrl @ negctrl @ y r0, r1, r2, r3, a1;
ctrl @ ctrl @ ctrl @ ctrl @ y r0, r1, r2, r3, a2;
negctrl @ ctrl @ ctrl @ ctrl @ y r0, r1, r2, r3, a3;
