function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%Computing cost function J
predictions = X * Theta';
diff = (predictions - Y).^2;
J= 1/2 * sum(sum(diff.*R)) + lambda/2 * sum(sum(Theta.^2)) + lambda/2 * sum(sum(X.^2));

%Computing gradients
for movie = 1 : num_movies
    valid_users = find(R(movie,:)==1); %Captures the users who have rated this 'movie'
    Theta_temp = Theta(valid_users,:); %Captures theta vector of all the users who have rated 'movie' : #valid_users x num_features
    Y_temp = Y(movie, valid_users); %Captures the actual rating which 'valid_users' having given to 'movie' : 1 x #valid_users
    diff = X(movie,:) * Theta_temp' - Y_temp; % 1 x #valid_users
    X_grad(movie,:) = diff * Theta_temp; % 1 x #valid_users * #valid_users x num_features = 1 x num_features 
    X_grad(movie,:) = X_grad(movie,:) + lambda * X(movie,:);
end

for user = 1 : num_users
    valid_movies = find(R(:,user)==1); %Captures the movies which are rated by 'user'
    X_temp = X(valid_movies,:); %Captures the X vector of all the movie which have been rated by 'user' : #valid_movies x num_features
    Y_temp = Y(valid_movies, user); %Captures the actual ratings of all the movies which 'user' has rated : #valid_movies x 1
    diff = X_temp * Theta(user,:)' - Y_temp; % #valid_movies x 1
    Theta_grad(user,:) = (X_temp' * diff)'; % num_features x #valid_movies * #valid_movies x 1
    Theta_grad(user,:) = Theta_grad(user,:) + lambda * Theta(user,:);
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
