function hiddenLayerBayes_16x16
clear all;

load 'labelDataSubset1_Train.mat';%Goes from 0 to 9
load 'pixelDataSubset1_Train.mat';
load 'labelDataSubset1_Test.mat';
load 'pixelDataSubset1_Test.mat';

numChars = 10;
charWidth = 16; charHeight = 16;
numHiddenNodes = charWidth*charHeight/4;
numPixelNodes = charWidth*charHeight;
numChildren = numPixelNodes/numHiddenNodes;
numTrainingSamples = size(pixelDataSubset_Train,1);
numTestingSamples = size(pixelDataSubset_Test,1);
numSamplesLimitTrain = 10000;%Per digit
numSamplesLimitTest = 1000;

if(0)
    
    pixelDataSubset_Train2 = zeros(numTrainingSamples,numPixelNodes);
    for i = 1:numTrainingSamples
        pixelDataSubset_Train2(i,:) = reshape(imresize(reshape(pixelDataSubset_Train(i,:),[28 28])',[charWidth charHeight])',[1 numPixelNodes]);
%         resizem(pixelDataSubset_Train,[size(pixelDataSubset_Train,1) numPixelNodes]);
    end
    pixelDataSubset_Train = pixelDataSubset_Train2;
    
    pixelDataSubset_Test2 = zeros(numTestingSamples,numPixelNodes);
    for i = 1:numTestingSamples
        pixelDataSubset_Test2(i,:) = reshape(imresize(reshape(pixelDataSubset_Test(i,:),[28 28])',[charWidth charHeight])',[1 numPixelNodes]);
%         resizem(pixelDataSubset_Test,[size(pixelDataSubset_Test,1) numPixelNodes]);
    end
    pixelDataSubset_Test = pixelDataSubset_Test2;
    %Threshold the pixel values
    pixelDataSubset_Train = (pixelDataSubset_Train > 127).*1;%Scale to 255 for display
    pixelDataSubset_Test = (pixelDataSubset_Test > 127).*1;
    
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
    save('pixelDataSubset1_TrainRearrange_16x16.mat','pixelDataSubset_TrainNew');
    save('pixelDataSubset1_TestRearrange_16x16.mat','pixelDataSubset_TestNew');
else
    load 'pixelDataSubset1_TrainRearrange_16x16.mat';
    load 'pixelDataSubset1_TestRearrange_16x16.mat';
end
numNodes = size(pixelDataSubset_TrainNew,2);
CPD_Hidden = zeros(numChars, numHiddenNodes, 2);%binary-valued hidden state
CPD_Hidden_Count = zeros(numChars, numHiddenNodes, 2);
CPD_Pixel = zeros(2, numNodes, 2);
CPD_Pixel_Count = zeros(2, numNodes, 2);%num of parent states=2
weightVct = zeros(numHiddenNodes,2^(numChildren+1),numChars);%zeros(numChars, numHiddenNodes, 2);
numEMIters = 500;%2000;
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
    reminder = mod(iter,100);
    if(reminder == 0)
        CPD_string = sprintf('CPD_Pixel_EM_16x16_%d.mat',iter);
        CPD_Pixel_EM = CPD_Pixel;
        save(CPD_string,'CPD_Pixel_EM');
        CPD_string = sprintf('CPD_Hidden_EM_16x16_%d.mat',iter);
        CPD_Hidden_EM = CPD_Hidden;
        save(CPD_string,'CPD_Hidden_EM');
    end
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
% fprintf('Starting testing (exact)... ');
% keyboard;
% tic;
% ConfusionMatrix2 = exactInference(pixelDataSubset_TestNew, labelDataSubset_Test, CPD_Pixel, CPD_Hidden);
% t1 = toc;
% fprintf('%g sec\r\n',t1);
% for i = 1:numChars
%     fprintf('%d: Accuracy: %g\r\t',i-1,ConfusionMatrix2(i,i)/sum(ConfusionMatrix2(i,:)));
%     for j = 1:numChars
%         fprintf('%d:%d; ',j-1,ConfusionMatrix2(i,j));
%     end
%     fprintf('\r');
% end
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
% charWidth = 28; charHeight = 28;
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