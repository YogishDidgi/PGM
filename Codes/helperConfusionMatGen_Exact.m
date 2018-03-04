numMatrices = 20;
numHiddenNodes = 16;
ConfusionMatrix_EM_8x8_EMCheck_exact = zeros(10,10,numHiddenNodes,numMatrices);
labelNodeSamples = randi(10,numInferenceIters,1);
hiddenNodeRandValues = rand(numInferenceIters,1);
for i = 1:1:numMatrices
    fprintf('Running %d ... ',i*1000);
    tic;
    string = sprintf('CPD_Hidden_EM_8x8_new_%d.mat',i*1000);load(string);
    string = sprintf('CPD_Pixel_EM_8x8_new_%d.mat',i*1000);load(string);
%     ConfusionMatrix_EM_8x8(:,:,i) = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);
%     ConfusionMatrix_EM_8x8_EMCheck(:,:,i) = approxInference_EMCheck(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters, labelNodeSamples, hiddenNodeRandValues);
    ConfusionMatrix_EM_8x8_EMCheck_exact(:,:,:,i) = exactInference_EMCheckPerHiddenNode(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM);
    t1 = toc;
    fprintf('%g sec\r',t1);
end
save('ConfusionMatrix_EM_8x8_EMCheck_exact.mat','ConfusionMatrix_EM_8x8_EMCheck_exact');
numLabels = 10;
numSamples = 1000;
accuracy = zeros(numHiddenNodes, numMatrices);
for hiddenNodeID = 1:numHiddenNodes
    for i = 1:numLabels
        accuracy(hiddenNodeID,:) = accuracy(hiddenNodeID,:) + squeeze(ConfusionMatrix_EM_8x8_EMCheck_exact(i,i,hiddenNodeID,:))'/numSamples;
    end
    accuracy(hiddenNodeID,:) = accuracy(hiddenNodeID,:)/numLabels;
end
for hiddenNodeID = 1:numHiddenNodes
    fprintf('HiddenNode: %d\r',hiddenNodeID);
    for i = 1:numMatrices
        fprintf('%g, ',accuracy(hiddenNodeID,i));
    end
    fprintf('\r');
end