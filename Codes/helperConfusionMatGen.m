numMatrices = 20;
ConfusionMatrix_EM_8x8_EMCheck = zeros(10,10,numMatrices);
labelNodeSamples = randi(10,numInferenceIters,1);
hiddenNodeRandValues = rand(numInferenceIters,1);
for i = 1:1:numMatrices
    fprintf('Running %d ... ',i*1000);
    tic;
    string = sprintf('CPD_Hidden_EM_8x8_new_%d.mat',i*1000);load(string);
    string = sprintf('CPD_Pixel_EM_8x8_new_%d.mat',i*1000);load(string);
%     ConfusionMatrix_EM_8x8(:,:,i) = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);
    ConfusionMatrix_EM_8x8_EMCheck(:,:,i) = approxInference_EMCheck(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters, labelNodeSamples, hiddenNodeRandValues);
    t1 = toc;
    fprintf('%g sec\r',t1);
end
save('ConfusionMatrix_EM_8x8_EMCheck.mat','ConfusionMatrix_EM_8x8_EMCheck');
numLabels = 10;
numSamples = 100;
accuracy = zeros(numMatrices,1);
for i = 1:numLabels
    accuracy = accuracy + squeeze(ConfusionMatrix_EM_8x8_EMCheck(i,i,:))/numSamples;
end
accuracy = accuracy/numLabels;
for i = 1:numMatrices
    fprintf('%d: %g\r',i,accuracy(i));
end

% keyboard;
% 
% i = 4000;fprintf('Running %d ...\r',i);
% string = sprintf('CPD_Hidden_EM_8x8_%d.mat',i);load(string);
% string = sprintf('CPD_Pixel_EM_8x8_%d.mat',i);load(string);
% tic;ConfusionMatrix_4000 = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);toc;
% 
% i = 5000;fprintf('Running %d ...\r',i);
% string = sprintf('CPD_Hidden_EM_8x8_%d.mat',i);load(string);
% string = sprintf('CPD_Pixel_EM_8x8_%d.mat',i);load(string);
% tic;ConfusionMatrix_5000 = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);toc;
% 
% i = 6000;fprintf('Running %d ...\r',i);
% string = sprintf('CPD_Hidden_EM_8x8_%d.mat',i);load(string);
% string = sprintf('CPD_Pixel_EM_8x8_%d.mat',i);load(string);
% tic;ConfusionMatrix_6000 = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);toc;
% 
% i = 7000;fprintf('Running %d ...\r',i);
% string = sprintf('CPD_Hidden_EM_8x8_%d.mat',i);load(string);
% string = sprintf('CPD_Pixel_EM_8x8_%d.mat',i);load(string);
% tic;ConfusionMatrix_7000 = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);toc;
% 
% i = 8000;fprintf('Running %d ...\r',i);
% string = sprintf('CPD_Hidden_EM_8x8_%d.mat',i);load(string);
% string = sprintf('CPD_Pixel_EM_8x8_%d.mat',i);load(string);
% tic;ConfusionMatrix_8000 = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);toc;
% 
% i = 9000;fprintf('Running %d ...\r',i);
% string = sprintf('CPD_Hidden_EM_8x8_%d.mat',i);load(string);
% string = sprintf('CPD_Pixel_EM_8x8_%d.mat',i);load(string);
% tic;ConfusionMatrix_9000 = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);toc;
% 
% i = 10000;fprintf('Running %d ...\r',i);
% string = sprintf('CPD_Hidden_EM_8x8_%d.mat',i);load(string);
% string = sprintf('CPD_Pixel_EM_8x8_%d.mat',i);load(string);
% tic;ConfusionMatrix_10000 = approxInference(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM, numInferenceIters);toc;
