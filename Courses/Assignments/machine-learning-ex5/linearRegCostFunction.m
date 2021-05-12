function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Computing J
h = theta' * X'; % 1 x 2 * 2 x m = 1xm
j = sum((h' - y).^2)/(2*m); % 1 x 1
reg_term = (lambda / (2*m)) * sum(theta(2:end,:).^2); % 1 x 1
J = j + reg_term;

%Computing grad
grad = ((h-y') * X) / m; % 1 x m * m x n = 1 * (n+1) 
reg_term = (lambda / m) * theta(2:end,:); % n x 1

grad = grad + [0 reg_term'];
% =========================================================================

grad = grad(:);

end
