function 4DOF_kane
SolveOrdinaryDifferentialEquations
% File  4DOF_kane.m  created by Autolev 4.1 on Tue May 26 21:14:56 2020


%===========================================================================
function VAR = ReadUserInput
global   Lax Lb Lt uAp uBp uCp uDp q1pp q2pp q3pp q4pp xpp ypp zpp;
global   q1 q2 q3 q4 uA uB uC uD x y z;
global   q1p q2p q3p q4p xp yp zp;
global   DEGtoRAD RADtoDEG;
global   TINITIAL TFINAL INTEGSTP PRINTINT ABSERR RELERR;

%-------------------------------+--------------------------+-------------------+-----------------
% Quantity                      | Value                    | Units             | Description
%-------------------------------|--------------------------|-------------------|-----------------
Lax                             =  0.0;                    % UNITS               Constant
Lb                              =  0.0;                    % UNITS               Constant
Lt                              =  0.0;                    % UNITS               Constant
uAp                             =  0.0;                    % UNITS               Constant
uBp                             =  0.0;                    % UNITS               Constant
uCp                             =  0.0;                    % UNITS               Constant
uDp                             =  0.0;                    % UNITS               Constant
q1pp                            =  0.0;                    % UNITS               Constant
q2pp                            =  0.0;                    % UNITS               Constant
q3pp                            =  0.0;                    % UNITS               Constant
q4pp                            =  0.0;                    % UNITS               Constant
xpp                             =  0.0;                    % UNITS               Constant
ypp                             =  0.0;                    % UNITS               Constant
zpp                             =  0.0;                    % UNITS               Constant

q1                              =  0.0;                    % UNITS               Initial Value
q2                              =  0.0;                    % UNITS               Initial Value
q3                              =  0.0;                    % UNITS               Initial Value
q4                              =  0.0;                    % UNITS               Initial Value
uA                              =  0.0;                    % UNITS               Initial Value
uB                              =  0.0;                    % UNITS               Initial Value
uC                              =  0.0;                    % UNITS               Initial Value
uD                              =  0.0;                    % UNITS               Initial Value
x                               =  0.0;                    % UNITS               Initial Value
y                               =  0.0;                    % UNITS               Initial Value
z                               =  0.0;                    % UNITS               Initial Value

TINITIAL                        =  0.0;                    % sec                 Initial Time
TFINAL                          =  1.0;                    % sec                 Final Time
INTEGSTP                        =  0.1;                    % sec                 Integration Step
PRINTINT                        =  1;                      % Positive Integer    Print-Integer
ABSERR                          =  1.0E-08;                %                     Absolute Error
RELERR                          =  1.0E-07 ;               %                     Relative Error
%-------------------------------+--------------------------+-------------------+-----------------

% Unit conversions
Pi       = 3.141592653589793;
DEGtoRAD = Pi/180.0;
RADtoDEG = 180.0/Pi;

% Evaluate constants
% Set the initial values of the states
VAR(1) = q1;
VAR(2) = q2;
VAR(3) = q3;
VAR(4) = q4;
VAR(5) = uA;
VAR(6) = uB;
VAR(7) = uC;
VAR(8) = uD;
VAR(9) = x;
VAR(10) = y;
VAR(11) = z;



%===========================================================================
function OpenOutputFilesAndWriteHeadings
FileIdentifier = fopen('4DOF_kane.1', 'wt');   if( FileIdentifier == -1 ) error('Error: unable to open file 4DOF_kane.1'); end
fprintf( 1,             '%%       t\n' );
fprintf( 1,             '%%     (sec)\n\n' );
fprintf(FileIdentifier, '%% FILE: 4DOF_kane.1\n%%\n' );
fprintf(FileIdentifier, '%%       t\n' );
fprintf(FileIdentifier, '%%     (sec)\n\n' );
FileIdentifier = fopen('4DOF_kane.2', 'wt');   if( FileIdentifier == -1 ) error('Error: unable to open file 4DOF_kane.2'); end
fprintf(FileIdentifier, '%% FILE: 4DOF_kane.2\n%%\n' );
fprintf(FileIdentifier, '%%       x              y\n' );
fprintf(FileIdentifier, '%%     (mm)           (mm)\n\n' );
FileIdentifier = fopen('4DOF_kane.3', 'wt');   if( FileIdentifier == -1 ) error('Error: unable to open file 4DOF_kane.3'); end
fprintf(FileIdentifier, '%% FILE: 4DOF_kane.3\n%%\n' );
fprintf(FileIdentifier, '%%      x''             y''\n' );
fprintf(FileIdentifier, '%%    (mm/s)         (mm/s)\n\n' );
FileIdentifier = fopen('4DOF_kane.4', 'wt');   if( FileIdentifier == -1 ) error('Error: unable to open file 4DOF_kane.4'); end
fprintf(FileIdentifier, '%% FILE: 4DOF_kane.4\n%%\n' );
fprintf(FileIdentifier, '%%      uA             uB             uC             uD\n' );
fprintf(FileIdentifier, '%%    (rad/s)        (rad/s)        (rad/s)        (rad/s)\n\n' );
FileIdentifier = fopen('4DOF_kane.5', 'wt');   if( FileIdentifier == -1 ) error('Error: unable to open file 4DOF_kane.5'); end
fprintf(FileIdentifier, '%% FILE: 4DOF_kane.5\n%%\n' );
fprintf(FileIdentifier, '%%      uA''            uB''            uC''            uD''\n' );
fprintf(FileIdentifier, '%%   (rad/s^2)      (rad/s^2)      (rad/s^2)      (rad/s^2)\n\n' );



%===========================================================================
% Main driver loop for numerical integration of differential equations
%===========================================================================
function SolveOrdinaryDifferentialEquations
global   Lax Lb Lt uAp uBp uCp uDp q1pp q2pp q3pp q4pp xpp ypp zpp;
global   q1 q2 q3 q4 uA uB uC uD x y z;
global   q1p q2p q3p q4p xp yp zp;
global   DEGtoRAD RADtoDEG;
global   TINITIAL TFINAL INTEGSTP PRINTINT ABSERR RELERR;

OpenOutputFilesAndWriteHeadings
VAR = ReadUserInput;
OdeMatlabOptions = odeset( 'RelTol',RELERR, 'AbsTol',ABSERR, 'MaxStep',INTEGSTP );
T = TINITIAL;
PrintCounter = 0;
mdlDerivatives(T,VAR,0);
while 1,
  if( TFINAL>=TINITIAL & T+0.01*INTEGSTP>=TFINAL ) PrintCounter = -1; end
  if( TFINAL<=TINITIAL & T+0.01*INTEGSTP<=TFINAL ) PrintCounter = -1; end
  if( PrintCounter <= 0.01 ),
     mdlOutputs(T,VAR,0);
     if( PrintCounter == -1 ) break; end
     PrintCounter = PRINTINT;
  end
  [TimeOdeArray,VarOdeArray] = ode45( @mdlDerivatives, [T T+INTEGSTP], VAR, OdeMatlabOptions, 0 );
  TimeAtEndOfArray = TimeOdeArray( length(TimeOdeArray) );
  if( abs(TimeAtEndOfArray - (T+INTEGSTP) ) >= abs(0.001*INTEGSTP) ) warning('numerical integration failed'); break;  end
  T = TimeAtEndOfArray;
  VAR = VarOdeArray( length(TimeOdeArray), : );
  PrintCounter = PrintCounter - 1;
end
mdlTerminate(T,VAR,0);



%===========================================================================
% mdlDerivatives: Calculates and returns the derivatives of the continuous states
%===========================================================================
function sys = mdlDerivatives(T,VAR,u)
global   Lax Lb Lt uAp uBp uCp uDp q1pp q2pp q3pp q4pp xpp ypp zpp;
global   q1 q2 q3 q4 uA uB uC uD x y z;
global   q1p q2p q3p q4p xp yp zp;
global   DEGtoRAD RADtoDEG;
global   TINITIAL TFINAL INTEGSTP PRINTINT ABSERR RELERR;

% Update variables after integration step
q1 = VAR(1);
q2 = VAR(2);
q3 = VAR(3);
q4 = VAR(4);
uA = VAR(5);
uB = VAR(6);
uC = VAR(7);
uD = VAR(8);
x = VAR(9);
y = VAR(10);
z = VAR(11);
q1p = uA;
q2p = uB;
q3p = uC;
q4p = uD;
xp = Lt*(sin(q3)*cos(q1)*cos(q4)-sin(q1)*(sin(q2)*sin(q4)-cos(q2)*cos(q3)*cos(q4)))*(uD+cos(q3)*uB+sin(q2)*sin(q3)*uA) - Lax*sin(q1)*uA - Lb*sin(q1)*uA - Lt*(cos(q1)*cos(q3)-sin(q1)*sin(q3)*cos(q2))*(sin(q3)*cos(q4)*uB-sin(q4)*uC-(sin(q4)*cos(q2)+sin(q2)*cos(q3)*cos(q4))*uA);
yp = Lax*cos(q1)*uA + Lb*cos(q1)*uA + Lt*(sin(q1)*sin(q3)*cos(q4)+cos(q1)*(sin(q2)*sin(q4)-cos(q2)*cos(q3)*cos(q4)))*(uD+cos(q3)*uB+sin(q2)*sin(q3)*uA) - Lt*(sin(q1)*cos(q3)+sin(q3)*cos(q1)*cos(q2))*(sin(q3)*cos(q4)*uB-sin(q4)*uC-(sin(q4)*cos(q2)+sin(q2)*cos(q3)*cos(q4))*uA);
zp = Lt*(sin(q2)*sin(q3)*(sin(q4)*uC-sin(q3)*cos(q4)*uB)-(sin(q4)*cos(q2)+sin(q2)*cos(q3)*cos(q4))*(uD+cos(q3)*uB));

% Update derivative array prior to integration step
VARp(1) = q1p;
VARp(2) = q2p;
VARp(3) = q3p;
VARp(4) = q4p;
VARp(5) = uAp;
VARp(6) = uBp;
VARp(7) = uCp;
VARp(8) = uDp;
VARp(9) = xp;
VARp(10) = yp;
VARp(11) = zp;

sys = VARp';



%===========================================================================
% mdlOutputs: Calculates and return the outputs
%===========================================================================
function Output = mdlOutputs(T,VAR,u)
global   Lax Lb Lt uAp uBp uCp uDp q1pp q2pp q3pp q4pp xpp ypp zpp;
global   q1 q2 q3 q4 uA uB uC uD x y z;
global   q1p q2p q3p q4p xp yp zp;
global   DEGtoRAD RADtoDEG;
global   TINITIAL TFINAL INTEGSTP PRINTINT ABSERR RELERR;

% Evaluate output quantities
Output(1)=T;
Output(2)=x*1000;  Output(3)=y*1000;
Output(4)=xp*1000;  Output(5)=yp*1000;
Output(6)=uA;  Output(7)=uB;  Output(8)=uC;  Output(9)=uD;
Output(10)=uAp;  Output(11)=uBp;  Output(12)=uCp;  Output(13)=uDp;
FileIdentifier = fopen('all');
WriteOutput( 1,                 Output(1:1) );
WriteOutput( FileIdentifier(1), Output(1:1) );
WriteOutput( FileIdentifier(2), Output(2:3) );
WriteOutput( FileIdentifier(3), Output(4:5) );
WriteOutput( FileIdentifier(4), Output(6:9) );
WriteOutput( FileIdentifier(5), Output(10:13) );



%===========================================================================
function WriteOutput( fileIdentifier, Output )
numberOfOutputQuantities = length( Output );
if numberOfOutputQuantities > 0,
  for i=1:numberOfOutputQuantities,
    fprintf( fileIdentifier, ' %- 14.6E', Output(i) );
  end
  fprintf( fileIdentifier, '\n' );
end



%===========================================================================
% mdlTerminate: Perform end of simulation tasks and set sys=[]
%===========================================================================
function sys = mdlTerminate(T,VAR,u)
FileIdentifier = fopen('all');
fclose( FileIdentifier(1) );
fclose( FileIdentifier(2) );
fclose( FileIdentifier(3) );
fclose( FileIdentifier(4) );
fclose( FileIdentifier(5) );
fprintf( 1, '\n Output is in the files 4DOF_kane.i  (i=1, ..., 5)\n' );
fprintf( 1, ' The output quantities and associated files are listed in the file 4DOF_kane.dir\n' );
sys = [];



%===========================================================================
% Sfunction: System/Simulink function from standard template
%===========================================================================
function [sys,x0,str,ts] = Sfunction(t,x,u,flag)
switch flag,
  case 0,  [sys,x0,str,ts] = mdlInitializeSizes;    % Initialization of sys, initial state x0, state ordering string str, and sample times ts
  case 1,  sys = mdlDerivatives(t,x,u);             % Calculate the derivatives of continuous states and store them in sys
  case 2,  sys = mdlUpdate(t,x,u);                  % Update discrete states x(n+1) in sys
  case 3,  sys = mdlOutputs(t,x,u);                 % Calculate outputs in sys
  case 4,  sys = mdlGetTimeOfNextVarHit(t,x,u);     % Return next sample time for variable-step in sys
  case 9,  sys = mdlTerminate(t,x,u);               % Perform end of simulation tasks and set sys=[]
  otherwise error(['Unhandled flag = ',num2str(flag)]);
end



%===========================================================================
% mdlInitializeSizes: Return the sizes, initial state VAR, and sample times ts
%===========================================================================
function [sys,VAR,stateOrderingStrings,timeSampling] = mdlInitializeSizes
sizes = simsizes;             % Call simsizes to create a sizes structure
sizes.NumContStates  = 11;    % sys(1) is the number of continuous states
sizes.NumDiscStates  = 0;     % sys(2) is the number of discrete states
sizes.NumOutputs     = 13;    % sys(3) is the number of outputs
sizes.NumInputs      = 0;     % sys(4) is the number of inputs
sizes.DirFeedthrough = 1;     % sys(6) is 1, and allows for the output to be a function of the input
sizes.NumSampleTimes = 1;     % sys(7) is the number of samples times (the number of rows in ts)
sys = simsizes(sizes);        % Convert it to a sizes array
stateOrderingStrings = [];
timeSampling         = [0 0]; % m-by-2 matrix containing the sample times
OpenOutputFilesAndWriteHeadings
VAR = ReadUserInput
