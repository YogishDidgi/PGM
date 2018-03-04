function naiveBayes
clear all;

load 'labelDataSubset_Train.mat';%Goes from 0 to 9
load 'pixelDataSubset_Train.mat';
load 'labelDataSubset_Test.mat';
load 'pixelDataSubset_Test.mat';

%Threshold the pixel values
pixelDataSubset_Train = (pixelDataSubset_Train > 127).*1;%Scale to 255 for display
pixelDataSubset_Test = (pixelDataSubset_Test > 127).*1;

numChars = 10;
numTrainingSamples = size(pixelDataSubset_Train,1);
numTestingSamples = size(pixelDataSubset_Test,1);
numNodes = size(pixelDataSubset_Train,2);
numSamplesLimitTrain = 1000;%Per digit
numSamplesLimitTest = 100;

CPD_Count = zeros(numChars, numNodes);
CPD = zeros(numChars, numNodes, 2);

fprintf('Starting training ...\r');
%ML estimate
for i = 1:numTrainingSamples
    CPD_Count(labelDataSubset_Train(i)+1,:) = CPD_Count(labelDataSubset_Train(i)+1,:) + pixelDataSubset_Train(i,:);
end
%Those whose values were 0, assign them to be a very low non-zero value;
%uniform is giving low accuracy
CPD_Count(find(CPD_Count == 0)) = 10;%numSamplesLimitTrain/2;
CPD(:,:,2) = CPD_Count/numSamplesLimitTrain;
CPD(:,:,1) = 1 - CPD(:,:,2);

fprintf('Starting testing ...\r');
AccuracyCount = zeros(numChars,2);
ConfusionMatrix = zeros(numChars,numChars);
%Classification
for i = 1:numTestingSamples
    for class = 1:numChars
        pbty(class,:) = pixelDataSubset_Test(i,:).*CPD(class,:,2) + (1 - pixelDataSubset_Test(i,:)).*CPD(class,:,1);
    end
    %pbtyLog = log(pbty);
    %jointPbty = sum(pbtyLog,2);
    jointPbty = prod(pbty,2);
    [value, classID] = max(jointPbty);
    if((classID-1) == labelDataSubset_Test(i))
        AccuracyCount(labelDataSubset_Test(i)+1,1) = AccuracyCount(labelDataSubset_Test(i)+1,1) + 1;
    else
        AccuracyCount(labelDataSubset_Test(i)+1,2) = AccuracyCount(labelDataSubset_Test(i)+1,2) + 1;
    end
    ConfusionMatrix(labelDataSubset_Test(i)+1,classID) = ConfusionMatrix(labelDataSubset_Test(i)+1,classID) + 1;
end
Accuracy = AccuracyCount/numSamplesLimitTest;
fprintf('Accuracy ...\r');
for class = 1:numChars
    fprintf('Char: %d\tCorrect: %d\tIncorrect: %d\tPercent: %g\r',class-1,AccuracyCount(class,1),AccuracyCount(class,2),Accuracy(class));
end

%Confusion matrix
for i = 1:numChars
    fprintf('%d:\r\t',i-1);
    for j = 1:numChars
        fprintf('%d:%d; ',j-1,ConfusionMatrix(i,j));
    end
    fprintf('\r');
end
keyboard;
end