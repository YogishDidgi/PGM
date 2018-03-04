function getData
clear all;
%Train data
%Label
fileID = fopen('..\data\train-labels.idx1-ubyte');
headerData = fread(fileID,2,'uint32','b');
fprintf('%d\t%d\r',headerData(1),headerData(2));
labelData_Train = fread(fileID,headerData(2),'uint8','b');
save('labelData_Train.mat','labelData_Train');
fclose(fileID);
%Pixel
fileID = fopen('..\data\train-images.idx3-ubyte');
headerData = fread(fileID,4,'uint32','b');
fprintf('%d\t%d\r',headerData(1),headerData(2));
pixelData_Train = uint8(zeros(headerData(2),headerData(3)*headerData(4)));
for i = 1:headerData(2)
    pixelData_Train(i,:) = fread(fileID,headerData(3)*headerData(4),'uint8','b');
end
save('pixelData_Train.mat','pixelData_Train');
fclose(fileID);

%Test data
%Label
fileID = fopen('..\data\t10k-labels.idx1-ubyte');
headerData = fread(fileID,2,'uint32','b');
fprintf('%d\t%d\r',headerData(1),headerData(2));
labelData_Test = fread(fileID,headerData(2),'uint8','b');
save('labelData_Test.mat','labelData_Test');
fclose(fileID);
%Pixel
fileID = fopen('..\data\t10k-images.idx3-ubyte');
headerData = fread(fileID,4,'uint32','b');
fprintf('%d\t%d\r',headerData(1),headerData(2));
pixelData_Test = uint8(zeros(headerData(2),headerData(3)*headerData(4)));
for i = 1:headerData(2)
    pixelData_Test(i,:) = fread(fileID,headerData(3)*headerData(4),'uint8','b');
end
save('pixelData_Test.mat','pixelData_Test');
fclose(fileID);
keyboard;
end