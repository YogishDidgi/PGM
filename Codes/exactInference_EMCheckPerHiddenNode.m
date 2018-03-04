%Perform exact inference and return the confusion matrix
%Sum over all possible values of hidden nodes for exact inference
function confusionMat = exactInference_EMCheckPerHiddenNode(pixelDataSubset_TrainNew, labelDataSubset_Train, CPD_Pixel_EM, CPD_Hidden_EM)
% CPD_Hidden = zeros(numChars, numHiddenNodes, 2);%binary-valued hidden state
% CPD_Pixel = zeros(2, numNodes, 2);
numPixelNodes = size(CPD_Pixel_EM,2);
numHiddenNodes = size(CPD_Hidden_EM,2);
numChildren = numPixelNodes/numHiddenNodes;
numChars = 10;
numSamples = size(pixelDataSubset_TrainNew,1);
confusionMat = zeros(numChars,numChars,numHiddenNodes);
parentHiddenNode = zeros(numPixelNodes);
for i = 1:numPixelNodes
	parentHiddenNode(i) = floor((i-1)/numHiddenNodes) + 1;
end
for sampleID = 1:numSamples
    for hiddenNodeID = 1:numHiddenNodes
        jointPbty = zeros(numChars,1);
        tempPbty_0 = ones(numChars,1);
        tempPbty_1 = ones(numChars,1);
        for char = 1:numChars
            tempPbty_0(char) = tempPbty_0(char)*CPD_Hidden_EM(char,hiddenNodeID,1);
            tempPbty_1(char) = tempPbty_1(char)*CPD_Hidden_EM(char,hiddenNodeID,2);
            for i = 1:numChildren
                tempPbty_0(char) = tempPbty_0(char)*CPD_Pixel_EM(1,(hiddenNodeID-1)*numChildren+i,pixelDataSubset_TrainNew(sampleID,(hiddenNodeID-1)*numChildren+i)+1);
                tempPbty_1(char) = tempPbty_1(char)*CPD_Pixel_EM(2,(hiddenNodeID-1)*numChildren+i,pixelDataSubset_TrainNew(sampleID,(hiddenNodeID-1)*numChildren+i)+1);
            end
        end
        total = tempPbty_0 + tempPbty_1;
        jointPbty = total/sum(total);
        [val,ID] = max(jointPbty);
        confusionMat(labelDataSubset_Train(sampleID)+1,ID,hiddenNodeID) = confusionMat(labelDataSubset_Train(sampleID)+1,ID,hiddenNodeID) + 1;
    end
end
end