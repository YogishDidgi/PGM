function confusionMat = exactInference(pixelDataSubset_Test, labelDataSubset_Test, CPD_Pixel, CPD_Hidden)
% CPD_Hidden = zeros(numChars, numHiddenNodes, 2);%binary-valued hidden state
% CPD_Pixel = zeros(2, numNodes, 2);
numPixelNodes = size(CPD_Pixel,2);
numHiddenNodes = size(CPD_Hidden,2);
numChildren = numPixelNodes/numHiddenNodes;
numChars = 10;
numSamples = size(pixelDataSubset_Test,1);
confusionMat = zeros(numChars,numChars);
parentHiddenNode = zeros(numPixelNodes);
for i = 1:numPixelNodes
	parentHiddenNode(i) = floor((i-1)/numChildren) + 1;
end
numStates = 2^numHiddenNodes;
for sampleID = 1:numSamples
    jointPbty = zeros(numChars,1);
    stateHidden = zeros(numHiddenNodes,1);
    tic;
    for stateID = 0:numStates-1
        tempPbty = ones(numChars,1);
        state = dec2bin(stateID, numHiddenNodes);%Get the state in binary for all nodes
        for i = 1:numHiddenNodes
            stateHidden(i) = str2double(state(i));%State in numeric type instead of ascii format
            for char = 1:numChars
                tempPbty(char) = tempPbty(char)*CPD_Hidden(char,i,stateHidden(i)+1);
            end
        end
        for i = 1:numPixelNodes
            tempPbty = tempPbty*CPD_Pixel(stateHidden(parentHiddenNode(i))+1,i,pixelDataSubset_Test(sampleID,i)+1);
        end
        jointPbty = jointPbty + tempPbty;
    end
    t1 = toc;
    fprintf('Sample %d of %d in %g sec\r',sampleID,numSamples,t1);
    jointPbty = jointPbty/sum(jointPbty);
    [val,ID] = max(jointPbty);
    confusionMat(labelDataSubset_Test(sampleID)+1,ID) = confusionMat(labelDataSubset_Test(sampleID)+1,ID) + 1;
end

end