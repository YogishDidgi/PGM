%function displayDigits
clear all;
sum1 = 0;
sum2 = 0;
H(:,:,1) = [88	0	2	1	0	1	3	0	3	2
1	95	0	3	0	1	0	0	0	0
22	5	49	14	1	0	1	3	4	1
8	2	2	76	0	2	2	3	3	2
9	2	1	1	47	2	4	3	4	27
21	1	0	22	3	26	2	3	19	3
16	4	5	1	2	5	65	0	2	0
19	10	0	2	1	1	0	58	2	7
14	3	1	12	4	3	0	2	55	6
2	2	1	2	10	0	0	16	2	65];
H(:,:,2) = [88	1	3	0	0	0	5	0	3	0
0	97	0	2	0	1	0	0	0	0
24	4	55	9	1	0	2	0	4	1
11	4	4	73	1	1	0	5	1	0
8	2	3	0	50	2	3	3	2	27
17	1	1	28	2	27	3	5	13	3
18	4	7	1	1	3	64	0	2	0
20	8	0	2	1	0	0	58	5	6
11	3	2	12	2	3	1	2	61	3
1	3	0	3	9	0	0	17	2	65];
H(:,:,3) = [85	0	1	1	0	1	7	0	4	1
0	91	0	3	0	3	0	0	2	1
4	4	60	12	1	2	4	1	11	1
2	3	2	81	0	3	0	3	2	4
3	0	0	3	64	2	3	2	3	20
10	4	2	22	2	32	1	4	12	11
15	3	1	0	3	4	70	0	4	0
4	6	1	3	0	0	1	71	1	13
5	4	2	16	0	6	0	1	59	7
1	1	0	3	6	0	0	4	2	83];
H(:,:,4) = [88	0	1	0	0	1	2	0	6	2
0	96	0	1	0	2	1	0	0	0
5	3	63	12	0	0	5	2	8	2
2	4	1	80	1	2	0	5	2	3
4	1	2	2	55	1	5	3	3	24
7	5	0	23	3	33	3	2	13	11
14	4	2	0	3	5	68	0	4	0
0	5	1	1	4	0	0	76	0	13
6	5	2	11	3	6	0	3	61	3
0	0	0	3	3	0	0	8	3	83];
H(:,:,5) = [56	1	4	1	4	8	15	2	9	0
58	27	1	3	3	0	0	2	1	5
6	9	50	7	6	3	6	3	9	1
9	8	3	58	7	5	0	7	1	2
16	7	1	0	34	1	5	14	2	20
17	2	0	12	6	27	0	21	6	9
19	2	6	2	9	2	52	0	6	2
16	10	0	1	17	4	0	24	5	23
15	8	6	10	2	4	6	12	32	5
21	9	1	0	15	1	0	18	0	35];
H(:,:,6) = [57	1	3	1	4	10	14	3	7	0
59	27	1	2	2	0	0	2	2	5
6	10	52	6	6	2	6	3	8	1
9	8	3	56	8	6	0	7	1	2
16	6	1	0	33	2	5	17	3	17
18	2	0	14	6	25	1	20	6	8
19	2	6	2	11	2	50	0	6	2
15	10	0	1	16	4	0	24	5	25
14	8	5	10	3	4	7	11	32	6
22	9	1	0	17	1	0	18	0	32];
H(:,:,7) = [55	2	1	2	8	9	12	2	9	0
0	61	1	3	23	0	0	5	1	6
1	12	51	6	11	3	6	3	6	1
3	12	3	58	10	7	0	4	1	2
1	8	1	0	48	1	5	18	2	16
9	4	0	13	12	27	0	18	5	12
13	2	6	2	17	2	50	0	6	2
0	17	0	1	23	4	0	30	4	21
9	8	6	10	7	4	6	11	31	8
1	15	1	0	29	1	0	20	0	33];

numMatrices = 7;
numLabels = 10;
numSamples = 100;
accuracy = zeros(numMatrices,1);
for i = 1:numLabels
    accuracy = accuracy + squeeze(H(i,i,:))/numSamples;
end
accuracy = accuracy/numLabels;
for matID = 1:numMatrices
    fprintf('Mat: %d:\r',matID);
    for i = 1:numLabels
        fprintf('%g\r',H(i,i,matID)/numSamples);
    end
    fprintf('\tAverage Accuracy:%d: %g\r',matID,accuracy(matID));
    keyboard;
end

%%
%{

load 'pixelDataSubset_Train.mat';%Goes from 0 to 9
%Threshold the pixel values
pixelDataSubset_TrainThres = (pixelDataSubset_Train > 127).*255;%Scale to 255 for display

numTrainingSamples = size(pixelDataSubset_TrainThres,1);
numSamplesLimitTrain = 1000;%Per digit
numNodes = size(pixelDataSubset_Train,2);
charWidth = sqrt(numNodes); charHeight = sqrt(numNodes);

figure,
for i = 1:10
    imageBefore = reshape(pixelDataSubset_Train((i-1)*numSamplesLimitTrain + 1,:),[charWidth charHeight])';
    imageAfter = reshape(pixelDataSubset_TrainThres((i-1)*numSamplesLimitTrain + 1,:),[charWidth charHeight])';
    subplot(5,4,2*i-1),imshow(imageBefore);
    subplot(5,4,2*i),imshow(imageAfter);
end
%}
keyboard;

