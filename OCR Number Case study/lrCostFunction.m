function [J, grad] = lrCostFunction(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
first_term_error_sum = (-y' * log(h));
second_term_error_sum = ((1-y') * log(1-h));
error_sum_without_regularization = (first_term_error_sum - second_term_error_sum) / m;
theta(1) = 0;
regularization_sum = (theta' * theta) * (lambda/(2 * m));
J = error_sum_without_regularization + regularization_sum;

unregularizated_gradient = (X' * (h - y)) / m;
regularization_term = theta * (lambda / m);
grad = unregularizated_gradient + regularization_term;




% =============================================================

end
