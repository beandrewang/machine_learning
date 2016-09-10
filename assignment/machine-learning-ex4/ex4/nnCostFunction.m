function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% do the forward propagation and get the final output
A1 = [ones(m, 1), X]; %  add ones generate the inputs
Z2 = A1 * Theta1';	% 5000x401 * 401x25 = 5000x25
A2 = sigmoid(Z2); 	% 5000x25
A2 = [ones(m, 1), A2]; 	% 5000x26
Z3 = A2 * Theta2';	% 5000x26 * 26x10 = 5000x10
A3 = sigmoid(Z3);	% 5000x10
h = A3;	% the output layer for all sets

% convert the vector y to the matrix
for i = 1 : m;
	Y(i, :) = zeros(1, num_labels);
	Y(i, y(i)) = 1;
end

% generate the cost func
for i = 1 : m
	for k = 1 : num_labels
		J = J + (-Y(i, k)) * log(h(i, k)) - (1 - Y(i, k)) * log(1 - h(i, k));
	end
end

% regularized part
regular1 = 0;

% the for loop version
for j = 1 : hidden_layer_size
	for k = 2 : input_layer_size + 1
		regular1 = regular1 + Theta1(j, k) ^ 2;
	end
end

regular2 = 0;
for j = 1 : num_labels
	for k = 2 : hidden_layer_size + 1
		regular2 = regular2 + Theta2(j, k) ^ 2;
	end
end

J = 1 / m * J + (lambda / 2 / m) * (regular1 + regular2);

% implement the back propagation algorithm
for t = 1 : m
	% step 1
	a1 = A1(t, :)';	% 401 x 1
	z2 = Z2(t, :)';	% 26 x 1
	a2 = A2(t, :)';	% 26 x 1
	z3 = Z3(t, :)';	% 10 x 1
	a3 = A3(t, :)';	% 10 x 1
	
	% step 2
	delta3 = a3 - Y(t, :)'; % 10 x 1
	
	% step 3
	delta2 = Theta2(:, 2 : end)' * delta3 .* sigmoidGradient(z2);  	% 25 x 10 * 10 x 1 .* 25 x 1 = 25 x 1
	
	% step 4
	Theta2_grad = Theta2_grad + delta3 * a2';	% 10 x 1 * 1 x 26 = 10 x 26
	Theta1_grad = Theta1_grad + delta2 * a1';	% 25 x 1 * 1 * 401 = 25 x 401
end

% step 5
Theta1_grad = (1 / m) * Theta1_grad;	% 25 x 401
Theta2_grad = (1 / m) * Theta2_grad; % 10 x 26

% regularization the gradient
regular1_grad = lambda / m * Theta1(:, 2 : end);
regular1_grad = [zeros(size(Theta1, 1), 1), regular1_grad];
regular2_grad = lambda / m * Theta2(:, 2 : end);
regular2_grad = [zeros(size(Theta2, 1), 1), regular2_grad];

Theta1_grad = Theta1_grad + regular1_grad;
Theta2_grad = Theta2_grad + regular2_grad;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
