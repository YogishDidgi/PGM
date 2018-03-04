function hiddenLayerBayes
clear all;

load 'labelDataSubset_Train.mat';%Goes from 0 to 9
load 'pixelDataSubset_Train.mat';
load 'labelDataSubset_Test.mat';
load 'pixelDataSubset_Test.mat';

numChars = 10;
charWidth = 4; charHeight = 4;
numHiddenNodes = charWidth*charHeight/4;
numPixelNodes = charWidth*charHeight;
numChildren = numPixelNodes/numHiddenNodes;
numTrainingSamples = size(pixelDataSubset_Train,1);
numTestingSamples = size(pixelDataSubset_Test,1);
numSamplesLimitTrain = 1000;%Per digit
numSamplesLimitTest = 100;

if(0)
    %Threshold the pixel values
    pixelDataSubset_Train = (pixelDataSubset_Train > 127).*1;%Scale to 255 for display
    pixelDataSubset_Test = (pixelDataSubset_Test > 127).*1;
    m=size(pixelDataSubset_Train,1);
    n=size(pixelDataSubset_Train,2);
    for i = 1:m
        pixelDataSubset_Train2(i,:)=interp1(pixelDataSubset_Train(i,:),1:52:n);
    end
    m=size(pixelDataSubset_Test,1);
    n=size(pixelDataSubset_Test,2);
    for i=1:m
        pixelDataSubset_Test2(i,:)=interp1(pixelDataSubset_Test(i,:),1:52:n);
    end
    pixelDataSubset_Train = pixelDataSubset_Train2;%resizem(pixelDataSubset_Train,[size(pixelDataSubset_Train,1) 16]);
    pixelDataSubset_Test = pixelDataSubset_Test2;%resizem(pixelDataSubset_Test,[size(pixelDataSubset_Test,1) 16]);
    
    %Construct dag
    % dag = constructDAG(numPixelNodes, numHiddenNodes);
    %Rearrange data for convenience
    pixelDataSubset_TrainNew = pixelDataSubset_Train;
    for i = 1:numTrainingSamples
        pixelDataSubset_TrainNew(i,:) = rearrangePixelData(pixelDataSubset_Train(i,:), numHiddenNodes,charWidth,charHeight);
    end
    pixelDataSubset_TestNew = pixelDataSubset_Test;
    for i = 1:numTestingSamples
        pixelDataSubset_TestNew(i,:) = rearrangePixelData(pixelDataSubset_Test(i,:), numHiddenNodes,charWidth,charHeight);
    end
    save('pixelDataSubset_TrainRearrange_4x4.mat','pixelDataSubset_TrainNew');
    save('pixelDataSubset_TestRearrange_4x4.mat','pixelDataSubset_TestNew');
else
    load 'pixelDataSubset_TrainRearrange_4x4.mat';
    load 'pixelDataSubset_TestRearrange_4x4.mat';
end
numNodes = size(pixelDataSubset_TrainNew,2);
CPD_Hidden = zeros(numChars, numHiddenNodes, 2);%binary-valued hidden state
CPD_Hidden_Count = zeros(numChars, numHiddenNodes, 2);
CPD_Pixel = zeros(2, numNodes, 2);
CPD_Pixel_Count = zeros(2, numNodes, 2);%num of parent states=2
weightVct = zeros(numHiddenNodes,2^(numChildren+1),numChars);%zeros(numChars, numHiddenNodes, 2);
numEMIters = 1000;
numInferenceIters = 10000;

% keyboard;
%Init-random
CPD_Hidden(:,:,1) = rand(numChars, numHiddenNodes);
CPD_Hidden(:,:,2) = 1 - CPD_Hidden(:,:,1);
CPD_Pixel(:,:,1) = rand(2, numNodes);
CPD_Pixel(:,:,2) = 1 - CPD_Pixel(:,:,1);

fprintf('Starting training ...\r');
for iter = 1:numEMIters
    fprintf('Iter: %d out of %d ... ', iter, numEMIters);
    tic;
    for i = 1:numTrainingSamples
        weightVct = EStep(pixelDataSubset_TrainNew(i,:),labelDataSubset_Train(i),CPD_Pixel,CPD_Hidden,weightVct);
    end
%     keyboard;
    [CPD_Pixel_Count,CPD_Pixel,CPD_Hidden_Count,CPD_Hidden] = MStep(weightVct,CPD_Pixel_Count,CPD_Pixel,CPD_Hidden_Count,CPD_Hidden);
    t1 = toc;
    fprintf('%g sec\r',t1);
end

fprintf('Starting testing ...\r');
% ConfusionMatrix = zeros(numChars,numChars);
ConfusionMatrix = approxInference(pixelDataSubset_TestNew, labelDataSubset_Test, CPD_Pixel, CPD_Hidden, numInferenceIters);
for i = 1:numChars
    fprintf('%d: Accuracy: %g\r\t',i-1,ConfusionMatrix(i,i)/sum(ConfusionMatrix(i,:)));
    for j = 1:numChars
        fprintf('%d:%d; ',j-1,ConfusionMatrix(i,j));
    end
    fprintf('\r');
end
keyboard;
end

function dag = constructDAG(numPixelNodes, numHiddenNodes)
widthPixel = sqrt(numPixelNodes);
heightPixel = sqrt(numPixelNodes);

widthHidden = sqrt(numHiddenNodes);
heightHidden = sqrt(numHiddenNodes);
numChildren = numPixelNodes/numHiddenNodes;
lengthChild = sqrt(numChildren);

dag = zeros(numPixelNodes+numHiddenNodes+1);

dag(1,2:numHiddenNodes+1) = 1;

for i = 1:numHiddenNodes
    quotient = floor((i-1)/widthHidden);
    reminder = mod(i-1,widthHidden);
    [X,Y] = ndgrid(lengthChild*quotient+1:lengthChild*quotient+lengthChild, ...
                    lengthChild*reminder+1:lengthChild*reminder+lengthChild);
    index = sub2ind([widthPixel,heightPixel],X,Y) + 1 + numHiddenNodes;%Offset for label and hidden nodes
    dag(i+1,index) = 1;
end
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





