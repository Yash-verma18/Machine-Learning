%% Entering Commands
% This example shows how to enter basic commands in MATLAB.
%
% Copyright 2018 The MathWorks, Inc.
%%
% To create and process variables, use the assignment operator '='.
disp('Assign value to variable m:')
disp('>> m = 2*1.5')
m = 2*1.5
%%
% A semicolon ';' to the end of a command suppresses the output.
disp('Add semicolon to suppress output:')
disp('>> m = 2*1.5;')
m = 2*1.5;
%%
% To conditionally execute statements, use the 'if', 'elseif', and 'else' statements.
disp(' ')
disp('Check value of m:')
if m > 5
    disp('m is greater than 5')
elseif m == 5
    disp('m is equal to 5')
else
    disp('m is less than 5')
end
%%
% MATLAB contains built-in functions and constants, such as 'abs' (absolute value) and 'pi'.
disp(' ')
disp('Calculate sine of pi/2, and assign it to y:')
disp('>> y = sin(pi/2)')
y = sin(pi/2) 