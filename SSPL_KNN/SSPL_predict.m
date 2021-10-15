function [Accuracy, predictLabel] = SSPL_predict(model, testData, testTarget, k, r)

testData = normr(testData);
[partialDataNeighbor, ~] = knnsearch(model.kdPartialData, testData, 'k', k);
[unlabelDataNeighbor, ~] = knnsearch(model.kdUnlabelData, testData, 'k', k);
%disp(size(partialDataNeighbor));
accNum = 0;
y1 = model.disambiguatedLabel;
y2 = model.pseudoLabel;
partialData = model.kdPartialData.X;
unlabelData = model.kdUnlabelData.X;    
[test_num, label_num] = size(testTarget);
predictLabel = zeros(test_num, label_num);

for i = 1:test_num
    partialNeighborIns = partialData(partialDataNeighbor(i,:),:)';
    unlabelNeighborIns = unlabelData(unlabelDataNeighbor(i,:),:)';
    w1 = lsqnonneg(partialNeighborIns, testData(i,:)');
    w2 = lsqnonneg(unlabelNeighborIns, testData(i,:)');
    partialNeighborLabel = y1(partialDataNeighbor(i,:), :);
    unlabelNeighborLabel = y2(unlabelDataNeighbor(i,:), :);
    minVal = inf;
    minIdx = -1;
    for label = 1:label_num
        tmp1 = partialNeighborLabel(:, label);
        tmp2 = unlabelNeighborLabel(:, label);
        restore = r * partialNeighborIns * (tmp1 .* w1) + (1 - r) * unlabelNeighborIns * (tmp2 .* w2);
        residual = norm(testData(i,:)' - restore, 2);
        %disp(size(restore));
        if residual < minVal
            minVal = residual;
            minIdx = label;
        end
    end
    predictLabel(i, minIdx) = 1;
    if testTarget(i, minIdx) == 1
        accNum = accNum + 1;
    end
end
Accuracy = accNum / test_num;