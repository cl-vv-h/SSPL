function res = knn_predict(test_data,X,Y,k)

a = size(test_data,1);
[b,l] = size(Y);

kdX = KDTreeSearcher(X);
[neighbor,~]=knnsearch(kdX, test_data,'k',k);
W = [];
for i=1:a
    neighborIns = X(neighbor(i,:),:)';
    w = lsqnonneg(neighborIns,test_data(i,:)');
    w = w';
    w_ = zeros(1,b);
    for j=1:k
        w_(1,neighbor(i,j)) = w(j);
    end
    W = [W;w_];
end
% disp(size(W));
% disp(size(Y));
su_res = W*Y;

[C,idx] = max(su_res,[],2);
res = zeros(a,l);
for i=1:a
    res(i,idx(i))=1;
end

end