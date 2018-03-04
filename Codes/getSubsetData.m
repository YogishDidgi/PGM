function getSubsetData
clear all;
load 'labelData_Train.mat';
load 'pixelData_Train.mat';
load 'labelData_Test.mat';
load 'pixelData_Test.mat';

numSamplesLimitTrain = 10;
numSamplesLimitTest = 1;

for i = 1:10
    %Train subset
    %Label
    index = find(labelData_Train == (i-1), numSamplesLimitTrain);
    labelDataSubset_Train(((i-1)*numSamplesLimitTrain + 1):(i*numSamplesLimitTrain),:) = labelData_Train(index);
    %Pixel
    pixelDataSubset_Train(((i-1)*numSamplesLimitTrain + 1):(i*numSamplesLimitTrain),:) = pixelData_Train(index,:);
    
    %Test subset
    %Label
    index = find(labelData_Test == (i-1), numSamplesLimitTest);
    labelDataSubset_Test(((i-1)*numSamplesLimitTest + 1):(i*numSamplesLimitTest),:) = labelData_Test(index);
    %Pixel
    pixelDataSubset_Test(((i-1)*numSamplesLimitTest + 1):(i*numSamplesLimitTest),:) = pixelData_Test(index,:);
end
save('labelDataSubset3_Train.mat','labelDataSubset_Train');
save('pixelDataSubset3_Train.mat','pixelDataSubset_Train');
save('labelDataSubset3_Test.mat','labelDataSubset_Test');
save('pixelDataSubset3_Test.mat','pixelDataSubset_Test');
keyboard;
end