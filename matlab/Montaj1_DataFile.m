% Simscape(TM) Multibody(TM) version: 6.1

% This is a model data file derived from a Simscape Multibody Import XML file using the smimport function.
% The data in this file sets the block parameter values in an imported Simscape Multibody model.
% For more information on this file, see the smimport function help page in the Simscape Multibody documentation.
% You can modify numerical values, but avoid any other changes to this file.
% Do not add code to this file. Do not edit the physical units shown in comments.

%%%VariableName:smiData


%============= RigidTransform =============%

%Initialize the RigidTransform structure array by filling in null values.
smiData.RigidTransform(9).translation = [0.0 0.0 0.0];
smiData.RigidTransform(9).angle = 0.0;
smiData.RigidTransform(9).axis = [0.0 0.0 0.0];
smiData.RigidTransform(9).ID = '';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(1).translation = [24.999999999999975 55 1.9320400064976285e-13];  % mm
smiData.RigidTransform(1).angle = 2.0943951023931975;  % rad
smiData.RigidTransform(1).axis = [0.57735026918962651 -0.57735026918962651 0.5773502691896244];
smiData.RigidTransform(1).ID = 'B[link0-1:-:link1-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(2).translation = [-50.000000000001265 85.000000000000085 -2.2382096176443156e-13];  % mm
smiData.RigidTransform(2).angle = 2.0943951023931997;  % rad
smiData.RigidTransform(2).axis = [-0.57735026918962706 -0.57735026918962706 -0.57735026918962296];
smiData.RigidTransform(2).ID = 'F[link0-1:-:link1-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(3).translation = [0 0 0.50000000000002132];  % mm
smiData.RigidTransform(3).angle = 3.141592653589786;  % rad
smiData.RigidTransform(3).axis = [1 -1.2346732204603792e-29 -3.4555691641457997e-15];
smiData.RigidTransform(3).ID = 'B[link1-1:-:link2-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(4).translation = [-8.5459417320521425e-14 0.4999999999993982 -1.0436096431476471e-13];  % mm
smiData.RigidTransform(4).angle = 2.0943951023931975;  % rad
smiData.RigidTransform(4).axis = [0.57735026918962651 -0.57735026918962529 0.57735026918962551];
smiData.RigidTransform(4).ID = 'F[link1-1:-:link2-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(5).translation = [-89.999999999999986 -33.000000000000014 0];  % mm
smiData.RigidTransform(5).angle = 2.0943951023931953;  % rad
smiData.RigidTransform(5).axis = [-0.57735026918962584 -0.57735026918962584 -0.57735026918962584];
smiData.RigidTransform(5).ID = 'B[link2-1:-:link3-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(6).translation = [100.00000000000007 7.1054273576010019e-14 -33.000000000000028];  % mm
smiData.RigidTransform(6).angle = 7.2217394619016807e-15;  % rad
smiData.RigidTransform(6).axis = [-0.98612192929621711 -0.16602271097987381 5.9116666733605509e-16];
smiData.RigidTransform(6).ID = 'F[link2-1:-:link3-1]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(7).translation = [-7.3200000000000074 -2.4980018054066022e-13 21.500000000000004];  % mm
smiData.RigidTransform(7).angle = 3.1415926535897865;  % rad
smiData.RigidTransform(7).axis = [1 0 0];
smiData.RigidTransform(7).ID = 'B[link3-1:-:link4-2]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(8).translation = [59.500000000000099 -9.3258734068513149e-15 21.499999999999815];  % mm
smiData.RigidTransform(8).angle = 3.1415926535897927;  % rad
smiData.RigidTransform(8).axis = [-1 -6.8626899533070693e-31 2.3338228759644266e-15];
smiData.RigidTransform(8).ID = 'F[link3-1:-:link4-2]';

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(9).translation = [-42.991207880023723 -50.93503916889636 0];  % mm
smiData.RigidTransform(9).angle = 0;  % rad
smiData.RigidTransform(9).axis = [0 0 0];
smiData.RigidTransform(9).ID = 'RootGround[link0-1]';


%============= Solid =============%
%Center of Mass (CoM) %Moments of Inertia (MoI) %Product of Inertia (PoI)

%Initialize the Solid structure array by filling in null values.
smiData.Solid(5).mass = 0.0;
smiData.Solid(5).CoM = [0.0 0.0 0.0];
smiData.Solid(5).MoI = [0.0 0.0 0.0];
smiData.Solid(5).PoI = [0.0 0.0 0.0];
smiData.Solid(5).color = [0.0 0.0 0.0];
smiData.Solid(5).opacity = 0.0;
smiData.Solid(5).ID = '';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(1).mass = 0.06692922562063508;  % kg
smiData.Solid(1).CoM = [46.10969828394002 -1.1456336397691402e-08 1.0008963644460712e-08];  % mm
smiData.Solid(1).MoI = [11.871709285303323 83.852324836838221 76.257627298923282];  % kg*mm^2
smiData.Solid(1).PoI = [-5.7244000474088646e-05 -2.9103947999352768e-08 2.9232595868917021e-08];  % kg*mm^2
smiData.Solid(1).color = [0.6470588235294118 0.51764705882352935 0];
smiData.Solid(1).opacity = 1;
smiData.Solid(1).ID = 'link3*:*Varsayilan';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(2).mass = 0.012605519241364601;  % kg
smiData.Solid(2).CoM = [40.580623258033995 0 0];  % mm
smiData.Solid(2).MoI = [0.65469460415198633 2.2345032609025304 2.2369625389016679];  % kg*mm^2
smiData.Solid(2).PoI = [0 0 0];  % kg*mm^2
smiData.Solid(2).color = [1 1 1];
smiData.Solid(2).opacity = 1;
smiData.Solid(2).ID = 'link4*:*Varsayilan';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(3).mass = 0.11599669781247939;  % kg
smiData.Solid(3).CoM = [-43.010835953883053 0 -9.4525122645003305e-05];  % mm
smiData.Solid(3).MoI = [22.810612624822305 120.61138061723017 119.88774588332666];  % kg*mm^2
smiData.Solid(3).PoI = [1.5644507923491882e-05 -0.0001188243449596174 0];  % kg*mm^2
smiData.Solid(3).color = [1 0.75686274509803919 0.56470588235294117];
smiData.Solid(3).opacity = 1;
smiData.Solid(3).ID = 'link2*:*Varsayilan';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(4).mass = 0.4534877614223074;  % kg
smiData.Solid(4).CoM = [2.3201931445457373 15.033863134715412 -0.41303631757785786];  % mm
smiData.Solid(4).MoI = [361.71461450258101 998.88167400598786 792.48053826931528];  % kg*mm^2
smiData.Solid(4).PoI = [0.76079372885185303 -12.359906754175212 -43.733207539496057];  % kg*mm^2
smiData.Solid(4).color = [0.77490196078431373 0.20705882352941174 0.20705882352941174];
smiData.Solid(4).opacity = 1;
smiData.Solid(4).ID = 'link0*:*Varsayilan';

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(5).mass = 0.29366423058628688;  % kg
smiData.Solid(5).CoM = [-30.83247000590881 50.474838618329642 -0.00056370176344830956];  % mm
smiData.Solid(5).MoI = [443.99402994388146 271.3924524669564 561.84384207938888];  % kg*mm^2
smiData.Solid(5).PoI = [0.001116261841950349 -0.00049697178598185027 183.3081130411806];  % kg*mm^2
smiData.Solid(5).color = [0.59999999999999998 0.59999999999999998 0.59999999999999998];
smiData.Solid(5).opacity = 1;
smiData.Solid(5).ID = 'link1*:*Varsayilan';


%============= Joint =============%
%X Revolute Primitive (Rx) %Y Revolute Primitive (Ry) %Z Revolute Primitive (Rz)
%X Prismatic Primitive (Px) %Y Prismatic Primitive (Py) %Z Prismatic Primitive (Pz) %Spherical Primitive (S)
%Constant Velocity Primitive (CV) %Lead Screw Primitive (LS)
%Position Target (Pos)

%Initialize the RevoluteJoint structure array by filling in null values.
smiData.RevoluteJoint(4).Rz.Pos = 0.0;
smiData.RevoluteJoint(4).ID = '';

smiData.RevoluteJoint(1).Rz.Pos = 12.757423611962333;  % deg
smiData.RevoluteJoint(1).ID = '[link0-1:-:link1-1]';

smiData.RevoluteJoint(2).Rz.Pos = -35.48692632684596;  % deg
smiData.RevoluteJoint(2).ID = '[link1-1:-:link2-1]';

smiData.RevoluteJoint(3).Rz.Pos = -173.0704264287869;  % deg
smiData.RevoluteJoint(3).ID = '[link2-1:-:link3-1]';

smiData.RevoluteJoint(4).Rz.Pos = 39.674684341909391;  % deg
smiData.RevoluteJoint(4).ID = '[link3-1:-:link4-2]';

