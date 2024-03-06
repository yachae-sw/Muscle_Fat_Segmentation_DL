function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,num)
% Partition CamVid data by randomly selecting 80% of the data for training.
% The rest is used for testing.
    
clear testIdx trainingImages testImages trainingLabels
clear trainingIdx testLabels
clear imdsTrain imdsTest pxdsTrain pxdsTest

% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
% shuffledIndices = randperm(numFiles);


% Use 20% of the images for testing.
testIdx = num:5:numFiles;

% Use the rest for training.
trainingIdx = setdiff(1:length(imds.Files),testIdx);

% Use 60% of the images for training.
% numTrain = round(0.60 * numFiles);
% trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
% numVal = round(0.20 * numFiles);
% valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
% testIdx = shuffledIndices(numTrain+numVal+1:end);

trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);

% Create image datastores for training and test.
imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = [255 170 85 000];

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);

end