% This is an examplar file on how the SSPL program could be used 
% (The main function is "SSPL_train.m" and "SSPL_predict.m")

% [1]Q.-W. Wang, Y.-F. Li, Z.-H. Zhou. Partial Label Learning with Unlabeled Data. In: Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI'19), Macau, China. 2019.
clear;
%load('sampleData.mat'); % Loading the file containing the necessary inputs for calling the SSPL function
load('sampleData.mat');
% Set the corresponding coefficients
k = 10;
r = 0.7;
alpha = 0.8;
beta = 0.25;
% train_target = train_p_target.';
% asd = zeros(20,38);
% test = test_target.'; 
model = SSPL_train(partialData, partialTarget, unlabeledData, k, alpha, beta);  % disambiguation phase
X_train = [partialData;unlabeledData];
Y_train = [model.disambiguatedLabel;model.pseudoLabel];
% model = SSPL_train(train_data, train_target, asd, k, alpha, beta);
% disp(sum(test,'all'));

% [accuracy, ~] = SSPL_predict(model, test_data, test, k, r);
total = size(testData,1);
max = 0;
cur = 0;

for iter = 1:20
res = knn_predict(testData, X_train,Y_train, k); 
true_lbl = testTarget;
count=0;
for i = 1:total
    if(res(i,:)==true_lbl(i,:))
        count = count+1;
    end
end
acc = count/total;
if(acc>max)
    max=acc;
    cur=iter;
end
%disp(acc);

end
disp(max);
disp(cur);

