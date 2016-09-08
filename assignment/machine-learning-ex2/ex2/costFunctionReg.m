function [J, grad] = costFunctionReg(theta, X, y, lambda)
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
featureNum = size(theta);

J1 = -y' * log(h);
J2 = -(1 - y)' * log(1 - h);
J3 = ones(1, featureNum - 1) * (theta(2 : end) .^ 2);

J = (1 / m) * (J1 + J2) + (lambda / 2 / m) * J3;

% calculate the gradients

grad(1) = (1 / m) * (h - y)' * X(:, 1);

for j = 2 : featureNum
		grad(j) = (1 / m) * (h - y)' * X(:, j) + lambda / m * theta(j);
end

% =============================================================

end
