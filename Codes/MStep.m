%%
%Function to compute Maximization
%weightVct - current weight count for the parameters
%thetaCount - current wieght count of the parameters
%thetaNormalized - current estimate of the parameters
%dag - Graph depicting connection between nodes

function [CPD_Pixel_Count,CPD_Pixel,CPD_Hidden_Count,CPD_Hidden] = MStep(weightVct,CPD_Pixel_Count,CPD_Pixel,CPD_Hidden_Count,CPD_Hidden)
numPixelNodes = size(CPD_Pixel,2);
numHiddenNodes = size(CPD_Hidden,2);
numChildren = numPixelNodes/numHiddenNodes;
numChars = size(weightVct,3);
%weightVct = zeros(numHiddenNodes,2^(numChildren+1),numChars);
[p,q,r]=size(weightVct);
for hiddenNodeID = 1:numHiddenNodes
    thetaNormalized{1} = [CPD_Hidden(:, hiddenNodeID, 1),...
                            CPD_Hidden(:, hiddenNodeID, 2)];
	thetaCount{1} = [CPD_Hidden_Count(:, hiddenNodeID, 1),...
                        CPD_Hidden_Count(:, hiddenNodeID, 2)];
    for i = 1:numChildren
        thetaNormalized{i+1} = [CPD_Pixel(:,(hiddenNodeID-1)*numChildren+i,1),...
                                CPD_Pixel(:,(hiddenNodeID-1)*numChildren+i,2)];
        thetaCount{i+1} = [CPD_Pixel_Count(:,(hiddenNodeID-1)*numChildren+i,1),...
                                CPD_Pixel_Count(:,(hiddenNodeID-1)*numChildren+i,2)];
    end
    dagNew = zeros(numChildren+1);
    dagNew(1,2:end) = 1;
%     weightVct(hiddenNodeID,:,sampleLabelData)
    [thetaCount, thetaNormalized] = MStep2(squeeze(weightVct(hiddenNodeID,:,:)), thetaCount, thetaNormalized, dagNew);
    CPD_Hidden(:, hiddenNodeID, 1) = thetaNormalized{1}(:,1);
    CPD_Hidden(:, hiddenNodeID, 2) = thetaNormalized{1}(:,2);
    CPD_Hidden_Count(:, hiddenNodeID, 1) = thetaCount{1}(:,1);
    CPD_Hidden_Count(:, hiddenNodeID, 2) = thetaCount{1}(:,2);
    for i = 1:numChildren
        CPD_Pixel(:,(hiddenNodeID-1)*numChildren+i,1) = thetaNormalized{i+1}(1,:)';
        CPD_Pixel(:,(hiddenNodeID-1)*numChildren+i,2) = thetaNormalized{i+1}(2,:)';
        CPD_Pixel_Count(:,(hiddenNodeID-1)*numChildren+i,1) = thetaCount{i+1}(1,:)';
        CPD_Pixel_Count(:,(hiddenNodeID-1)*numChildren+i,2) = thetaCount{i+1}(2,:)';
    end
end

end



function [thetaCount, thetaNormalized] = MStep2(weightVct, thetaCount, thetaNormalized, dag)
%weightVct = zeros(numHiddenNodes,2^(numChildren+1),numChars);
N = size(thetaNormalized, 1);%Number of nodes in network
stateNew = zeros(N,1);
%Iterate through each weight and assign that weight to corresponding CPD bin
for stateID = 0:size(weightVct,1)-1
    state = dec2bin(stateID, N);%Get the state in binary for all nodes
    for i = 1:N
        stateNew(i) = str2double(state(i));%State in numeric type instead of ascii format
    end
    %Iterate through each node and check parent state
    %Hidden node
    nodeID = 1;
    nodeState = stateNew(nodeID) + 1;
    thetaCount{nodeID}(:, nodeState) = thetaCount{nodeID}(:, nodeState) + weightVct(stateID + 1,:)';
    %Pixel nodes
    for nodeID = 2:N
        power = 0;
        parentState = 0;
        for i = N:-1:1
            if((i ~= nodeID) && dag(i,nodeID))%Check if parent exists
                %Multiply with the correct scale factor
                parentState = parentState + (2^power)*stateNew(i);
                power = power + 1;
            end
        end
        parentState = parentState + 1;
        nodeState = stateNew(nodeID) + 1;
        thetaCount{nodeID}(parentState, nodeState) = thetaCount{nodeID}(parentState, nodeState) + sum(weightVct(stateID + 1,:));
    end
end

%Calculate the new CPD parameters by normalizing
for nodeID = 1:N
    [rows,cols] = size(thetaCount{nodeID});
    for i = 1:rows
        thetaNormalized{nodeID}(i,:) = thetaCount{nodeID}(i,:)/sum(thetaCount{nodeID}(i,:));
    end
end
end