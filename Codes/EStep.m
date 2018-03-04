%%
%Function to compute Expectation
%sampleVct - current data sample
%missingSample - array indicating which nodes' values are missing
%thetaNormalized - current estimate of the parameters
%weightVct - current weight count for the parameters
%dag - Graph depicting connection between nodes

function weightVct = EStep(samplePixelData,sampleLabelData,CPD_Pixel,CPD_Hidden,weightVct)
% weightVct = zeros(numHiddenNodes,2^(numChildren+1),numChars);
% CPD_Hidden = zeros(numChars, numHiddenNodes, 2);%binary-valued hidden state
% CPD_Pixel = zeros(2, numNodes, 2);
numPixelNodes = size(CPD_Pixel,2);
numHiddenNodes = size(CPD_Hidden,2);
numChildren = numPixelNodes/numHiddenNodes;
missingSample = [1 zeros(1,numChildren)];
for hiddenNodeID = 1:numHiddenNodes
    thetaNormalized{1} = [CPD_Hidden(sampleLabelData+1, hiddenNodeID, 1),...
                            CPD_Hidden(sampleLabelData+1, hiddenNodeID, 2)];
    for i = 1:numChildren
        thetaNormalized{i+1} = [CPD_Pixel(:,(hiddenNodeID-1)*numChildren+i,1),...
                                CPD_Pixel(:,(hiddenNodeID-1)*numChildren+i,2)];
    end
    dagNew = zeros(numChildren+1);
    dagNew(1,2:end) = 1;
    sampleVct = [0 samplePixelData((hiddenNodeID-1)*numChildren+1:(hiddenNodeID-1)*numChildren+numChildren)];
    weightVct(hiddenNodeID,:,sampleLabelData+1) = EStep2(sampleVct, missingSample, thetaNormalized, weightVct(hiddenNodeID,:,sampleLabelData+1), dagNew);
end

end


function weightVct = EStep2(sampleVct, missingSample, thetaNormalized, weightVct, dag)
N = size(thetaNormalized, 1);%Number of nodes in network
numNodesMissing = sum(missingSample);%Number of missing nodes in sample
numStates = 2^numNodesMissing;

%The un-normalized probabilities will be computed using the product of
%conditional probabilites in the Markov Blanket
pbty = ones(numStates, 1);

for stateID = 0:numStates-1%Iterate through all the states
	state = dec2bin(stateID, numNodesMissing);%Get the state in binary for missing nodes
    stateFull = sampleVct;%Holds the state for all nodes
    index = 1;
    for i = 1:N
        if(missingSample(i))
            stateFull(i) = str2double(state(index));
            missingNodeID(index) = i;%Array containing ID for missing nodes
            index = index + 1;
        end
    end
    sampleConsideredFlag = zeros(N,1);
    for missingNode = 1:numNodesMissing
        %Check parent state
        power = 0;
        parentState = 0;
        for i = N:-1:1
            if((i ~= missingNodeID(missingNode)) && dag(i,missingNodeID(missingNode)))%Check if parent exists
                %Multiply with the correct scale factor
                parentState = parentState + (2^power)*stateFull(i);
                power = power + 1;
            end
        end
        parentState = parentState + 1;
        %Multiply the current node's conditional probability
        if(~sampleConsideredFlag(missingNodeID(missingNode)))
            pbty(stateID + 1) = pbty(stateID + 1)*thetaNormalized{missingNodeID(missingNode)}(parentState, stateFull(missingNodeID(missingNode)) + 1);
            sampleConsideredFlag(missingNodeID(missingNode)) = 1;
        end
        %Check children state
        for childID = 1:N
            if((childID ~= missingNodeID(missingNode)) && dag(missingNodeID(missingNode),childID) && ~sampleConsideredFlag(childID))%Check if child exists
                %Check child's parent state
                power = 0;
                parentState = 0;
                for i = N:-1:1
                    if((i ~= childID) && dag(i,childID))%Check if parent exists
                        parentState = parentState + (2^power)*stateFull(i);
                        power = power + 1;
                    end
                end
                parentState = parentState + 1;
                %Multiply the current node's children's conditional probability
                pbty(stateID + 1) = pbty(stateID + 1)*thetaNormalized{childID}(parentState, stateFull(childID) + 1);
                sampleConsideredFlag(childID) = 1;
            end
        end
    end
end

%Normalize the computed pbty values and assign the weights
pbty = pbty/sum(pbty);
for stateID = 0:numStates-1%Iterate through all the states
	state = dec2bin(stateID, numNodesMissing);%Get the state in binary for missing nodes
    stateFull = sampleVct;%Holds the state for all nodes
    index = 1;
    for i = 1:N
        if(missingSample(i))
            stateFull(i) = str2double(state(index));
            index = index + 1;
        end
    end
    %Iterate through all nodes and get correct index for weight vct
    power = 0;
    weightState = 0;
    for i = N:-1:1
        weightState = weightState + (2^power)*stateFull(i);
        power = power + 1;
    end
    weightState = weightState + 1;
    weightVct(weightState) = weightVct(weightState) + pbty(stateID + 1);
end

end
