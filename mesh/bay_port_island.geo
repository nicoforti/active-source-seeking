SetFactory("OpenCASCADE");

// ----------------------------
// Parameters
// ----------------------------
Lx = 6000;      // meters
Ly = 4000;

meshSize = 300; // target size in meters (120)

// Island
ix = 3500;
iy = 2200;
ir = 300;

// Port inlet (on right boundary)
portY0 = 1600;
portY1 = 2400;
portInset = 600;   // how deep the port "cut" goes left

// Bay indentation (top boundary dip)
bayX0 = 2200;
bayX1 = 4200;
bayDepth = 700;

// ----------------------------
// Outer boundary with bay + port
// Build a polygon (counter-clockwise)
// ----------------------------

// Points along bottom edge (left->right)
p1 = newp; Point(p1) = {0, 0, 0, meshSize};
p2 = newp; Point(p2) = {Lx, 0, 0, meshSize};

// Right edge with port cut:
// go up to portY0, then inset, then up to portY1, then out to right edge
p3 = newp; Point(p3) = {Lx, portY0, 0, meshSize};
p4 = newp; Point(p4) = {Lx - portInset, portY0, 0, meshSize};
p5 = newp; Point(p5) = {Lx - portInset, portY1, 0, meshSize};
p6 = newp; Point(p6) = {Lx, portY1, 0, meshSize};
p7 = newp; Point(p7) = {Lx, Ly, 0, meshSize};

// Top edge with bay indentation (right->left):
p8  = newp; Point(p8)  = {bayX1, Ly, 0, meshSize};
p9  = newp; Point(p9)  = {bayX1, Ly - bayDepth, 0, meshSize};
p10 = newp; Point(p10) = {bayX0, Ly - bayDepth, 0, meshSize};
p11 = newp; Point(p11) = {bayX0, Ly, 0, meshSize};
p12 = newp; Point(p12) = {0, Ly, 0, meshSize};

// Left edge
// back to p1

// Lines
l1 = newl; Line(l1) = {p1, p2};     // bottom
l2 = newl; Line(l2) = {p2, p3};     // right (bottom to port)
l3 = newl; Line(l3) = {p3, p4};     // port boundary segment (horizontal inward)
l4 = newl; Line(l4) = {p4, p5};     // inner port wall (vertical)
l5 = newl; Line(l5) = {p5, p6};     // port boundary segment (horizontal outward)
l6 = newl; Line(l6) = {p6, p7};     // right (port to top)
l7 = newl; Line(l7) = {p7, p8};     // top (right to bay start)
l8 = newl; Line(l8) = {p8, p9};     // bay down
l9 = newl; Line(l9) = {p9, p10};    // bay bottom
l10= newl; Line(l10)= {p10, p11};   // bay up
l11= newl; Line(l11)= {p11, p12};   // top (bay to left)
l12= newl; Line(l12)= {p12, p1};    // left edge

outerLoop = newll;
Line Loop(outerLoop) = {l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12};

// ----------------------------
// Island hole
// ----------------------------
Disk(1) = {ix, iy, 0, ir, ir};
islandCurves[] = Boundary{ Surface{1}; };
Printf("islandCurves = %g", islandCurves[]);

islandLoop = newll;
Line Loop(islandLoop) = { islandCurves[] };

// Domain surface with hole
Plane Surface(2) = { outerLoop, islandLoop };
Delete { Surface{1}; }

// ----------------------------
// Physical groups for boundaries
// ----------------------------

Physical Curve("open_sea_boundary") = {l12};
Physical Curve("port_boundary") = {l3, l5};

Physical Curve("island_boundary") = { islandCurves[] };

// Everything else on the outer boundary is coast.
Physical Curve("coast_boundary") = {l1,l2,l4,l6,l7,l8,l9,l10,l11};

Physical Surface("fluid_domain") = {2};

// Mesh options
Mesh.Algorithm = 6; // Frontal-Delaunay
Mesh.CharacteristicLengthMin = meshSize * 0.6;
Mesh.CharacteristicLengthMax = meshSize * 1.4;
