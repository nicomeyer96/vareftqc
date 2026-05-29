//d
//a0,a1
//r0,r1
OPENQASM 3;
qubit d;
qubit a0;
qubit a1;
qubit r0;
qubit r1;
// SYNDROME EXTRACTION
// initial hadamard layer
h r0;
h r1;
// first syndrome wire (stabilizer ZZI)
cz r0,d;
cz r0,a0;
// second syndrome wire (stabilizer IZZ)
cz r1,a0;
cz r1,a1;
// final hadamard layer
h r0;
h r1;
// RECOVERY (COHERENT)
ctrl @ negctrl @ x r0, r1, d;
ctrl @ ctrl @ x r0, r1, a0;
negctrl @ ctrl @ x r0, r1, a1;
