function hiddenLayerBayes_8x8_EMChanges
clear all;

load 'labelDataSubset1_Train.mat';%Goes from 0 to 9
load 'pixelDataSubset1_Train.mat';
load 'labelDataSubset1_Test.mat';
load 'pixelDataSubset1_Test.mat';
load 'pixelDataSubset1_TrainRearrange_8x8.mat';
load 'pixelDataSubset1_TestRearrange_8x8.mat';

numChars = 10;
charWidth = 8; charHeight = 8;
numHiddenNodes = charWidth*charHeight/4;
numPixelNodes = charWidth*charHeight;
numChildren = numPixelNodes/numHiddenNodes;
numTrainingSamples = size(pixelDataSubset_Train,1);
numTestingSamples = size(pixelDataSubset_Test,1);

CPD_Hidden = zeros(numChars, numHiddenNodes, 2);%binary-valued hidden state
CPD_Hidden_Count = zeros(numChars, numHiddenNodes, 2);
CPD_Pixel = zeros(2, numPixelNodes, 2);
CPD_Pixel_Count = zeros(2, numPixelNodes, 2);%num of parent states=2
weightVct = zeros(numHiddenNodes,2*numTrainingSamples);
numEMIters = 20000;%2000;
numInferenceIters = 10000;

% keyboard;
%Init-random
CPD_Hidden(:,:,1) = rand(numChars, numHiddenNodes);
CPD_Hidden(:,:,2) = 1 - CPD_Hidden(:,:,1);
CPD_Pixel(:,:,1) = rand(2, numPixelNodes);
CPD_Pixel(:,:,2) = 1 - CPD_Pixel(:,:,1);

fprintf('Starting training ...\r');
for iter = 1:numEMIters
    fprintf('Iter: %d out of %d ... ', iter, numEMIters);
    %{
    NOTE: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT
    1. Initialize counts everytime?
    2. Previous iteration weight is not used in the E-code
    %}
    CPD_Hidden_Count = zeros(numChars, numHiddenNodes, 2);
    CPD_Pixel_Count = zeros(2, numPixelNodes, 2);%num of parent states=2
    tic;
    weightVct = computeExpectation(pixelDataSubset_TrainNew,labelDataSubset_Train,CPD_Pixel,CPD_Hidden,weightVct);
    [CPD_Pixel_Count,CPD_Pixel,CPD_Hidden_Count,CPD_Hidden] = computeMaximization(pixelDataSubset_TrainNew,labelDataSubset_Train,...
                                                                                    weightVct,CPD_Pixel_Count,CPD_Pixel,CPD_Hidden_Count,CPD_Hidden);
    t1 = toc;
    fprintf('%g sec\r',t1);
%     reminder = mod(iter,1000);
%     if(reminder == 0)
%         CPD_string = sprintf('CPD_Pixel_EM_8x8_new_%d.mat',iter);
%         CPD_Pixel_EM = CPD_Pixel;
%         save(CPD_string,'CPD_Pixel_EM');
%         CPD_string = sprintf('CPD_Hidden_EM_8x8_new_%d.mat',iter);
%         CPD_Hidden_EM = CPD_Hidden;
%         save(CPD_string,'CPD_Hidden_EM');
%     end
end

fprintf('Starting testing %d iters (approx.)... ',numInferenceIters);
%keyboard;
% ConfusionMatrix = zeros(numChars,numChars);
tic;
ConfusionMatrix1 = approxInference(pixelDataSubset_TestNew, labelDataSubset_Test, CPD_Pixel, CPD_Hidden, numInferenceIters);
t1 = toc;
fprintf('%g sec\r\n',t1);
for i = 1:numChars
    fprintf('%d: Accuracy: %g\r\t',i-1,ConfusionMatrix1(i,i)/sum(ConfusionMatrix1(i,:)));
    for j = 1:numChars
        fprintf('%d:%d; ',j-1,ConfusionMatrix1(i,j));
    end
    fprintf('\r');
end
keyboard;
end

%Function to perform expectation of the hidden nodes, return the weight
%vector
function weightVct = computeExpectation(pixelDataSubset_TrainNew,labelDataSubset_Train,CPD_Pixel,CPD_Hidden,weightVct)
numTrainingSamples = size(pixelDataSubset_TrainNew,1);
numPixelNodes = size(CPD_Pixel,2);
numHiddenNodes = size(CPD_Hidden,2);
numChildren = numPixelNodes/numHiddenNodes;

for sampleID = 1:numTrainingSamples
    for hiddenNodeID = 1:numHiddenNodes
        %Compute the weight for two possible states of the hidden node
        pbty_0 = CPD_Hidden(labelDataSubset_Train(sampleID)+1,hiddenNodeID,1);
        pbty_1 = CPD_Hidden(labelDataSubset_Train(sampleID)+1,hiddenNodeID,2);
        for i = 1:numChildren
            pbty_0 = pbty_0*CPD_Pixel(1,(hiddenNodeID-1)*numChildren+i,pixelDataSubset_TrainNew(sampleID,(hiddenNodeID-1)*numChildren+i)+1);
            pbty_1 = pbty_1*CPD_Pixel(2,(hiddenNodeID-1)*numChildren+i,pixelDataSubset_TrainNew(sampleID,(hiddenNodeID-1)*numChildren+i)+1);
        end
        pbty_sum = pbty_0 + pbty_1;
        weightVct(hiddenNodeID,sampleID) = pbty_0/pbty_sum;%+ weightVct(hiddenNodeID,sampleID);
        weightVct(hiddenNodeID,sampleID+numTrainingSamples) = pbty_1/pbty_sum;%+ weightVct(hiddenNodeID,sampleID+numTrainingSamples);
    end
end
end

%Perform maximization; Compute the distribution parameters as a counting
%problem from the weight vector
function [CPD_Pixel_Count,CPD_Pixel,CPD_Hidden_Count,CPD_Hidden] = computeMaximization(pixelDataSubset_TrainNew,labelDataSubset_Train,...
                                                                                    weightVct,CPD_Pixel_Count,CPD_Pixel,CPD_Hidden_Count,CPD_Hidden)
numTrainingSamples = size(pixelDataSubset_TrainNew,1);
numPixelNodes = size(CPD_Pixel,2);
numHiddenNodes = size(CPD_Hidden,2);
numChildren = numPixelNodes/numHiddenNodes;

for sampleID = 1:numTrainingSamples
    for hiddenNodeID = 1:numHiddenNodes
        CPD_Hidden_Count(labelDataSubset_Train(sampleID)+1,hiddenNodeID,1) = CPD_Hidden_Count(labelDataSubset_Train(sampleID)+1,hiddenNodeID,1) ...
                                                                        + weightVct(hiddenNodeID,sampleID);
        CPD_Hidden_Count(labelDataSubset_Train(sampleID)+1,hiddenNodeID,2) = CPD_Hidden_Count(labelDataSubset_Train(sampleID)+1,hiddenNodeID,2) ...
                                                                        + weightVct(hiddenNodeID,sampleID+numTrainingSamples);
        for i = 1:numChildren
            CPD_Pixel_Count(1,(hiddenNodeID-1)*numChildren+i,pixelDataSubset_TrainNew(sampleID,(hiddenNodeID-1)*numChildren+i)+1) = ...
                CPD_Pixel_Count(1,(hiddenNodeID-1)*numChildren+i,pixelDataSubset_TrainNew(sampleID,(hiddenNodeID-1)*numChildren+i)+1) ...
                + weightVct(hiddenNodeID,sampleID);
            CPD_Pixel_Count(2,(hiddenNodeID-1)*numChildren+i,pixelDataSubset_TrainNew(sampleID,(hiddenNodeID-1)*numChildren+i)+1) = ...
                CPD_Pixel_Count(2,(hiddenNodeID-1)*numChildren+i,pixelDataSubset_TrainNew(sampleID,(hiddenNodeID-1)*numChildren+i)+1) ...
                + weightVct(hiddenNodeID,sampleID+numTrainingSamples);
        end
    end
end
for hiddenNodeID = 1:numHiddenNodes
    normalizationFactor = CPD_Hidden_Count(:,hiddenNodeID,1) + CPD_Hidden_Count(:,hiddenNodeID,2);
    CPD_Hidden(:,hiddenNodeID,1) = CPD_Hidden_Count(:,hiddenNodeID,1)./normalizationFactor;
    CPD_Hidden(:,hiddenNodeID,2) = CPD_Hidden_Count(:,hiddenNodeID,2)./normalizationFactor;
    for i = 1:numChildren
        normalizationFactor = CPD_Pixel_Count(:,(hiddenNodeID-1)*numChildren+i,1) + CPD_Pixel_Count(:,(hiddenNodeID-1)*numChildren+i,2);
        CPD_Pixel(:,(hiddenNodeID-1)*numChildren+i,1) = CPD_Pixel_Count(:,(hiddenNodeID-1)*numChildren+i,1)./normalizationFactor;
        CPD_Pixel(:,(hiddenNodeID-1)*numChildren+i,2) = CPD_Pixel_Count(:,(hiddenNodeID-1)*numChildren+i,2)./normalizationFactor;
    end
end
end