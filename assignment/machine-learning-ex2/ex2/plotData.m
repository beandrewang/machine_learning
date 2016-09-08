function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

traningSetSize = size(y);
negativeXIndex = 1;
positiveXIndex = 1;

for i = 1 : traningSetSize
		if(y(i) == 0) % the negative examples
				negativeX(negativeXIndex, :) = X(i, :);
				negativeXIndex = negativeXIndex + 1;
		elseif(y(i) == 1) % the positive examples
				positiveX(positiveXIndex, :) = X(i, :);
				positiveXIndex = positiveXIndex + 1;
		else
				fprintf('error, unexpected y\n');
				return;
		end
end

% plot the positiveX
plot(positiveX(:, 1), positiveX(:, 2), 'k+');

% plot the negativeX
plot(negativeX(:, 1), negativeX(:, 2), 'ko');
hold on;





% =========================================================================



hold off;

end
