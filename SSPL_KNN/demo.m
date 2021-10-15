% This is an examplar file on how the SSPL program could be used 
% (The main function is "SSPL_train.m" and "SSPL_predict.m")

% [1]Q.-W. Wang, Y.-F. Li, Z.-H. Zhou. Partial Label Learning with Unlabeled Data. In: Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI'19), Macau, China. 2019.

%load('sampleData.mat'); % Loading the file containing the necessary inputs for calling the SSPL function
load('sample data.mat');
% Set the corresponding coefficients
k = 15;
r = 0.7;
alpha = 0.8;
beta = 0.25;
train_target = train_p_target.';
asd = zeros(20,38);
test = test_target.';
%model = SSPL_train(partialData, partialTarget, unlabeledData, k, alpha, beta);  % disambiguation phase
model = SSPL_train(train_data, train_target, asd, k, alpha, beta);
disp(sum(test,'all'));
%[accuracy, ~] = SSPL_predict(model, testData, testTarget, k, r);                % testing phase
[accuracy, ~] = SSPL_predict(model, test_data, test, k, r);
fprintf('classification accuracy: %.3f\n', accuracy);
