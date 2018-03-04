function naiveBayes_2Layer
clear all;

load 'labelDataSubset_Train.mat';%Goes from 0 to 9
load 'pixelDataSubset_Train.mat';
load 'labelDataSubset_Test.mat';
load 'pixelDataSubset_Test.mat';

%Threshold the pixel values
pixelDataSubset_Train = (pixelDataSubset_Train > 127).*1;%Scale to 255 for display
pixelDataSubset_Test = (pixelDataSubset_Test > 127).*1;
% pixelDataSubset_Train = resizem(pixelDataSubset_Train,[size(pixelDataSubset_Train,1) 16]);
% pixelDataSubset_Test = resizem(pixelDataSubset_Test,[size(pixelDataSubset_Test,1) 16]);

numChars = 10;
numTrainingSamples = size(pixelDataSubset_Train,1);
numTestingSamples = size(pixelDataSubset_Test,1);
numNodes_L1 = size(pixelDataSubset_Train,2)/4;
numNodes_L2 = size(pixelDataSubset_Train,2);
numSamplesLimitTrain = 1000;%Per digit
numSamplesLimitTest = 100;
charWidth = sqrt(numNodes_L2); charHeight = sqrt(numNodes_L2);

CPD_Count_L1 = zeros(numChars, numNodes_L1);
CPD_L1 = zeros(numChars, numNodes_L1, 2);
CPD_Count_L2 = zeros(2, numNodes_L2);
CPD_L2 = zeros(2, numNodes_L2, 2);
numChildren = numNodes_L2/numNodes_L1;

fprintf('Starting training ...\r');
if(0)
    tic;
    pixelDataSubset_TrainNew = pixelDataSubset_Train;
    for i = 1:numTrainingSamples
        pixelDataSubset_TrainNew(i,:) = rearrangePixelData(pixelDataSubset_Train(i,:), numNodes_L1,charWidth,charHeight);
    end
    toc;
    tic;
    pixelDataSubset_TestNew = pixelDataSubset_Test;
    for i = 1:numTestingSamples
        pixelDataSubset_TestNew(i,:) = rearrangePixelData(pixelDataSubset_Test(i,:), numNodes_L1,charWidth,charHeight);
    end
    toc;
    save('pixelDataSubset_TrainRearrange.mat','pixelDataSubset_TrainNew');
    save('pixelDataSubset_TestRearrange.mat','pixelDataSubset_TestNew');
else
    load 'pixelDataSubset_TrainRearrange.mat';
    load 'pixelDataSubset_TestRearrange.mat';
end



pixelDataTrain_L1 = zeros(numTrainingSamples,numNodes_L1);%Hidden/Average of L2
pixelDataTrain_L2 = pixelDataSubset_TrainNew;%pixelDataSubset_Train;

%ML estimate
for i = 1:numTrainingSamples
    %Determine L1 nodes' values
    for j = 1:numNodes_L1
        pixelDataTrain_L1(i,j) = ceil(sum(pixelDataTrain_L2(i,(j-1)*numChildren + 1:(j-1)*numChildren + numChildren))/numChildren);
    end
    CPD_Count_L1(labelDataSubset_Train(i)+1,:) = CPD_Count_L1(labelDataSubset_Train(i)+1,:) + pixelDataTrain_L1(i,:);
    for j = 1:numNodes_L2
        parentID = floor((j-1)/numChildren)+1;
        CPD_Count_L2(pixelDataTrain_L1(i,parentID)+1,j) = CPD_Count_L2(pixelDataTrain_L1(i,parentID)+1,j) + pixelDataTrain_L2(i,j);
    end
end
%Those whose values were 0, assign them to be a very low non-zero value;
%uniform is giving low accuracy
CPD_Count_L1(find(CPD_Count_L1 == 0)) = 10;%numSamplesLimitTrain/2;
CPD_Count_L2(find(CPD_Count_L2 == 0)) = 10;%numSamplesLimitTrain/2;

CPD_L1(:,:,2) = CPD_Count_L1/numSamplesLimitTrain;
CPD_L1(:,:,1) = 1 - CPD_L1(:,:,2);
CPD_L2(:,:,2) = CPD_Count_L2/numSamplesLimitTrain;
CPD_L2(:,:,1) = 1 - CPD_L2(:,:,2);

fprintf('Starting testing ...\r');
AccuracyCount = zeros(numChars,2);
ConfusionMatrix = zeros(numChars,numChars);
%Classification
pixelDataTest_L1 = zeros(numTestingSamples,numNodes_L1);%Hidden/Average of L2
pixelDataTest_L2 = pixelDataSubset_TestNew;%pixelDataSubset_Test

for i = numTestingSamples:-1:1
    %Determine L1 nodes' values
    for j = 1:numNodes_L1
        pixelDataTest_L1(i,j) = ceil(sum(pixelDataTest_L2(i,(j-1)*numChildren + 1:(j-1)*numChildren + numChildren))/numChildren);
    end
    for class = 1:numChars
        pbty(class,:) = pixelDataTest_L1(i,:).*CPD_L1(class,:,2) + (1 - pixelDataTest_L1(i,:)).*CPD_L1(class,:,1);
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

function pixelDataNew = rearrangePixelData(pixelData, numHiddenNodes,charWidth,charHeight)
pixelDataNew = pixelData';
numPixelNodes = size(pixelDataNew,1);
numChildren = numPixelNodes/numHiddenNodes;
widthPixel = sqrt(numPixelNodes);
heightPixel = sqrt(numPixelNodes);

pixelDataMat = reshape(pixelData,charWidth,charHeight)';

widthHidden = sqrt(numHiddenNodes);
lengthChild = sqrt(numChildren);
index = 1;
for i = 1:numHiddenNodes
    quotient = floor((i-1)/widthHidden);
    reminder = mod(i-1,widthHidden);
    [X,Y] = ndgrid(lengthChild*quotient+1:lengthChild*quotient+lengthChild, ...
                    lengthChild*reminder+1:lengthChild*reminder+lengthChild);
	index = sub2ind([widthPixel,heightPixel],X,Y);
    pixelDataNew((i-1)*numChildren+1:(i-1)*numChildren+numChildren) = pixelData(index);
    index = index + 1;
end

end
