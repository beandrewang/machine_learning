function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

feature_size = size(X)(2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    theta_prev = theta;
		h = X * theta_prev; 
		for(j = 1 : feature_size)
				derivate_j = (h - y)' * X(:, j);
				theta(j) = theta(j) - (alpha / m) * derivate_j;
		end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
		if(iter > 1)
				if(J_history(iter) > J_history(iter - 1))
						fprintf('J is increasing, last %f, new %f\n', J_history(iter - 1), J_history(iter));
						return;
				end
		end
end

end
