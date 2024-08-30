clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%
% Hyper-parameters
%%%%%%%%%%%%%%%%%%%%%%%%%
learning_rate = 1e-4;
num_epochs = 100;
batch_size = 32;

%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%
% Training data
data_train = readmatrix('Dataset/train_house.csv');
X = data_train(:, 1:37);
y = data_train(:, 38);

% Test data
data_valid = readmatrix('Dataset/test_house.csv');
x_valid = data_valid(:, 1:37);
y_valid = data_valid(:, 38);

% Define the architecture of the neural network
input_size = 37;                          % Feature size
hidden_size1 = 128;                       % First hidden size
hidden_size2 = 64;                        % Second hidden size
output_size = 1;                          % Output size

% Initialize weights and biases randomly
W1 = randn(hidden_size1, input_size);   % First weights
b1 = zeros(hidden_size1, 1);            % First bias
W2 = randn(hidden_size2, hidden_size1); % Second weights
b2 = zeros(hidden_size2, 1);            % Second bias
W3 = randn(output_size, hidden_size2);  % Third weights
b3 = zeros(output_size, 1);             % Third bias

% Define the activation function (ReLU)
ReLU = @(x) max(0, x);
ReLU_deriv = @(x) x > 0;
% Define the activation function (sigmoid)
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_deriv = @(x) sigmoid(x) .* (1 - sigmoid(x));
% Define the activation function (tanh)
tanh_func = @(x) tanh(x);
tanh_deriv = @(x) 1 - tanh(x).^2;

% Define the loss function (Mean Squared Error)
loss = @(error) sum((error).^2);

%%%%%%%%%%%%%%%%%%%%%%%%%
% Training loop for SGD
%%%%%%%%%%%%%%%%%%%%%%%%%
tic
for epoch = 1:num_epochs

    for i = 1:batch_size:size(X, 1)
        batch_end = min(i + batch_size - 1, size(X, 1));

        % Forward pass
        x = X(i:batch_end, :)';
        y_true = y(i:batch_end)';

        z1 = W1 * x + b1;
        a1 = ReLU(z1);

        z2 = W2 * a1 + b2;
        a2 = ReLU(z2);

        z3 = W3 * a2 + b3;
        y_pred = z3;

        % Compute loss
        error =  y_pred - y_true;

        % Compute for backpropagation
        delta3 = 2/batch_size * (error);

        % Gradients for W3 and b3
        dW3 = delta3 * a2';
        db3 = sum(delta3, 2);

        % Backpropagate to second hidden layer
        delta2 = (W3' * delta3) .* ReLU_deriv(z2);

        % Gradients for W2 and b2
        dW2 = delta2 * a1';
        db2 = sum(delta2, 2);

        % Backpropagate to first hidden layer
        delta1 = (W2' * delta2) .* ReLU_deriv(z1);

        % Gradients for W1 and b1
        dW1 = delta1 * x';
        db1 = sum(delta1, 2);

        % Update weights and biases
        W3 = W3 - learning_rate * dW3;
        b3 = b3 - learning_rate * db3;

        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;

        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
    end

    % Print the loss every 10 epochs
    if mod(epoch, 10) == 0
        y_pred = W3 * ReLU(W2 * ReLU(W1 * x_valid' + b1) + b2) + b3;
        loss_valid = loss(y_valid'-y_pred) / size(y_valid, 1);
        fprintf('Epoch %d, Validation Loss: %.4f\n', epoch, loss_valid);
    end
end
toc
