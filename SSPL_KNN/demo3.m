% This is an examplar file on how the SSPL program could be used 
% (The main function is "SSPL_train.m" and "SSPL_predict.m")

% [1]Q.-W. Wang, Y.-F. Li, Z.-H. Zhou. Partial Label Learning with Unlabeled Data. In: Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI'19), Macau, China. 2019.

%load('sampleData.mat'); % Loading the file containing the necessary inputs for calling the SSPL function
load('d.mat');
% Set the corresponding coefficients
k = 10;
r = 0.7;
alpha = 0.7;
beta = 0.25;
% train_target = train_p_target.';
asd = zeros(20,34);
test = test_target.';
%model = SSPL_train(partialData, partialTarget, unlabeledData, k, alpha, beta);  % disambiguation phase
model = SSPL_train(double(train_data), train_label, asd, k, alpha, beta);
disp(sum(test,'all'));
%[accuracy, ~] = SSPL_predict(model, testData, testTarget, k, r);                % testing phase
[accuracy, ~] = SSPL_predict(model, double(test_data), test_label, k, r);
fprintf('classification accuracy: %.3f\n', accuracy);
