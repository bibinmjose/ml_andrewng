function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
%  neural network which performs classification
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

a1=[ones(m,1) X];   % m x (h+1)
z2=a1*Theta1';      % m x h
a2=sigmoid(z2);     % m x h
a2=[ones(m,1) a2];  % m x (h+1)
z3=a2*Theta2';      % m x k
h_tx=sigmoid(z3);   % m x k


for k=1:num_labels
        y_index=(y==k);
    J = J + -(1/m)*sum(y_index.*log(h_tx(:,k))+(1-y_index).*log(1-h_tx(:,k)));
end

J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));


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
D_3=zeros(num_labels,1);
% clear y_index;

for i=1:m
    for k=1:num_labels
        y_index=y(i)==k;
        D_3(k)=h_tx(i,k)-y_index;
    end
        a_2=sigmoidGradient([1, z2(i,:)]);
        
        D_2=(Theta2'*D_3).*a_2'; % 26 x 1
        D_2=D_2(2:end);
        
        %size(Theta2_grad), size(D_3), size(a2(i,:))
        
        Theta1_grad = Theta1_grad + D_2*a1(i,:);
        Theta2_grad = Theta2_grad + D_3*a2(i,:);

end


Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m).*Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m).*Theta2(:, 2:end);

 %fprintf('Size of Theta2:'),size(Theta2_grad),  size(D_3), size(a2)
 %size(Theta1_grad),  size(D_2), size(a1)

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
