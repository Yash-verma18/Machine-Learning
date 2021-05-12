function [J ,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J, grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%Modifying value of y

%For num_labels = 10, creates a 10x10 identity matrix
dummy_eye = eye(num_labels); 
%For each row i in y, the command Chooses ith row corresponding in dummy_eye
%eg: For i = 10,it spurs out [0,0,0,0,0,0,0,0,0,1] 
y = dummy_eye(y,:); % 5000 x 10

%Performing forward propagation
a_1 = [ones(size(X,1),1) X]; % 5000 x 401
z_2 = Theta1 * a_1';         % (25 x 401) * (401 x 5000) = 25 x 5000

a_2 = sigmoid(z_2); % 25 x 5000
a_2 = a_2'; % 5000 x 25; Transposing a_2 to maintain consistency
a_2 = [ones(size(a_2,1),1) a_2]; % 5000 x 26; Adding the bias term
z_3 = Theta2 * a_2'; % (10 x 26) * (26 x 5000) = 10 x 5000

h = sigmoid(z_3); % 10 x 5000
h = h'; %5000 x 10

%Finally computing cost function(unregularized)
inner = y .* log(h) + (1-y) .* log(1-h) ;% 5000 x 10
inner_sum = sum(inner, 2); % 5000 x 1; Sum of all columns, row wise
outer_sum = sum(inner_sum, 1); % 1 x 1; Sum of al row value,i.e. for each training example
J = -1/m * outer_sum; %Finally taking the mean over the number of training examples

%Computing the regularization term

%Since, we dont apply regularization to the bias terms, we need to reshape
%our theta matrices
reg_term = 0;

Theta1 = Theta1(:, 2:end); % 25 x 400
Theta2 = Theta2(:, 2:end); % 10 x 25

reg_1_inner= Theta1.^2; % 25 x 400
reg_1_innersum = sum(reg_1_inner,2); % 25 x 1
reg_1_outersum = sum(reg_1_innersum,1); % 1 x 1


reg_2_inner= Theta2.^2; % 10 x 25
reg_2_innersum = sum(reg_2_inner,2); % 10 x 1
reg_2_outersum = sum(reg_2_innersum,1); % 1 x 1

reg_term = ( lambda/(2*m) ) * (reg_1_outersum + reg_2_outersum);

%Finally, adding the regularization term to our cost function
J = J + reg_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta_3 = h - y; %5000 x 10 ; Has errors for the 3rd layer

delta_2_a = delta_3 * Theta2; %5000 x 10 * 10 x 25 = 5000 x 25
delta_2_b = sigmoidGradient(z_2'); % 5000 x 25

delta_2 = delta_2_a .* delta_2_b; % 5000 x 25

tri_delta_1 = delta_2' * a_1;% 25 x 5000 * 5000 x 401 = 25 x 401
tri_delta_2 = delta_3' * a_2;% 10 x 5000 * 5000 x 26 = 10 x 26

Theta1_grad = 1/m * tri_delta_1;
Theta2_grad = 1/m * tri_delta_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
