%Perform inference using likelihood weighted sampling
%Return the confusion matrix
function confusionMat = approxInference_EMCheck(pixelDataSubset_Test, labelDataSubset_Test, CPD_Pixel, CPD_Hidden, numIters, labelNodeSamples, hiddenNodeRandValues)
% CPD_Hidden = zeros(numChars, numHiddenNodes, 2);%binary-valued hidden state
% CPD_Pixel = zeros(2, numNodes, 2);
numPixelNodes = size(CPD_Pixel,2);
numHiddenNodes = size(CPD_Hidden,2);
numChildren = numPixelNodes/numHiddenNodes;
numChars = 10;
numSamples = size(pixelDataSubset_Test,1);
confusionMat = zeros(numChars,numChars);
for sampleID = 1:numSamples
    numerator = zeros(numChars,1);
    denominator = 0;
%     for label = 1:numChars
        for iter = 1:numIters
            weight = realmax;%Not one bcoz the value was losing precision because of large num of nodes
            hiddenNode = zeros(numHiddenNodes,1);
            %Sample label node
            %Sample hidden nodes one by one
            %Take pixel nodes as evidence
            label = labelNodeSamples(iter);%randi(numChars);
            for hiddenNodeID = 1:numHiddenNodes
                hiddenNode(hiddenNodeID) = hiddenNodeRandValues(iter) > CPD_Hidden(label,hiddenNodeID, 1);
%                 hiddenNode(hiddenNodeID) = rand > CPD_Hidden(label,hiddenNodeID, 1);
            end
            for i = 1:numPixelNodes
                parentHiddenNode = floor((i-1)/numChildren)+1;
                weight = weight*CPD_Pixel(hiddenNode(parentHiddenNode)+1,i,pixelDataSubset_Test(sampleID,i)+1);
            end
            denominator = denominator + weight;
            for i = 1:numChars
                numerator(i) = numerator(i) + weight*(label == i);%(labelDataSubset_Test(sampleID)+1));
            end
        end
        ratio = numerator/denominator;
%     end
    [val,ID] = max(ratio);
    confusionMat(labelDataSubset_Test(sampleID)+1,ID) = confusionMat(labelDataSubset_Test(sampleID)+1,ID) + 1;
end
end