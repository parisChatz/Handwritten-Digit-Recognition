function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   The parameters for the neural network are "unrolled" into the vector
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

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. 
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



%FORWARD PROPAGATION
%layer 1
  a1 = [ones(m, 1) X]; %bias
  
%layer 2
  z2 = a1*Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(m, 1) a2]; %bias
  
%layer 3
  z3 = a2*Theta2';
  a3 = sigmoid(z3);
  h = a3;

%Fix y matrix
%Output matrix y is not a matrix of size m,1. It is a matrix with size m,k
%and every column is the logic output of every classifier. For example :
%
%For output y=5,we want the 5 column and y(:,5) should be all zeros except
%the fives (5) that should be ones (1)
 
y = eye(num_labels)(y, :);

%cost function without regularization
J = sum(sum(((1/m)*(-y.*log(h)-(1-y).*log(1-h))))) ;

% REGULARIZATION
reg_term = (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) +  sum(sum(Theta2(:,2:end).^2)));
J = J + reg_term;

% BACK PROPOGATION
%errors
d3 = a3 - y;
d2 = (d3*Theta2).*[ones(size(z2,1),1) sigmoidGradient(z2)];
d2 = d2(:,2:end); % delta_1 is not calculated because we do not associate error with the input    

%Deltas
delta2 = d3'*a2; % has same dimensions as Theta2
delta1 = d2'*a1; % has same dimensions as Theta1
  
% Big delta update/ New Thetas
Theta1_grad = (1/m).*delta1;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m)*Theta1(:,2:end));

Theta2_grad = (1/m).*delta2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m)*Theta2(:,2:end));
  
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
