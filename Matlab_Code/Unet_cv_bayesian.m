clc;close all;clear;

%% load dataset(Image and Mask)

datasetpath = 'D:\yachae_sw\CTImages\segment_data\';
imageDir = fullfile(datasetpath,'image');
maskDir = fullfile(datasetpath,'mask');

imds = imageDatastore(imageDir);

% check image
% I = readimage(imds,51);
% imshow(I)

classes = [ "VAT", "Muscle","SAT","background"];

labelIDs   = [255 170 85 000];

pxds = pixelLabelDatastore(maskDir,classes,labelIDs);

% check muscle iamge
% C = readimage(pxds,51);
% B = labeloverlay(I,C,'IncludedLabels',"Muscle");
% imshow(B)

%% bayesian optimization

num_folds = 5;

for fold = 4 : num_folds

    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,fold);
    tbl = countEachLabel(pxdsTrain);
    frequency = tbl.PixelCount/sum(tbl.PixelCount);
    imageFreq = sqrt(tbl.PixelCount ./ tbl.ImagePixelCount);
    classWeightsSAT = imageFreq(4) ./ imageFreq(1);
    classWeightsVAT = imageFreq(4) ./ imageFreq(2);
    classWeightsMuscle = imageFreq(4) ./ imageFreq(3);
    classWeightsfactor = 1/min([classWeightsMuscle,classWeightsVAT,classWeightsSAT])*2;
    fprintf('Processing %d among %d folds \n', fold,5); % 5-fold cross validation

    optimVars = [
                optimizableVariable('L2Regularization',[0.00001 0.01],'Type','real')
                optimizableVariable('InitialLearnRate',[0.00001 0.01],'Type','real')
                optimizableVariable('miniBatchSize',[16 128],'Type','integer')
                optimizableVariable('GradientThreshold',[1 6],'Type','integer')
                optimizableVariable('GradientThresholdMethod',{'global-l2norm','l2norm'},'Type','categorical')];

                % optimizableVariable('classWeightfactor', [classWeightsfactor 1], 'Type', 'real')
                % optimizableVariable('classWeightSAT', [1 classWeightsSAT], 'Type', 'real')
                % optimizableVariable('classWeightVAT', [1 classWeightsVAT], 'Type', 'real')
                % optimizableVariable('classWeightMuscle', [1 classWeightsMuscle], 'Type', 'real')

    % bayesian object function
    ObjFcn = @(optimVars) UNETCVOpti(optimVars, imds, pxds, fold);

    BayesObjectR = bayesopt(ObjFcn,optimVars,...
        'AcquisitionFunctionName','expected-improvement-plus','PlotFcn',[],...
        'IsObjectiveDeterministic',false,...
        'MaxObjectiveEvaluations',50,...
        'UseParallel',false);

    % best parameter extraction
    bayesianopsegment{fold,1} = bestPoint(BayesObjectR);
    bayesianopsegment{fold,2} = BayesObjectR;
end

% save('bestParameterR_seg.mat', 'bayesianopsegment');


%% U-net model and 5-fold cross validation

test_count_st = 1;
test_count_end = 1;
load('pixelspace.mat');
result_model = struct('net',cell(1,num_folds));
result_space = struct('Name', cell(1, size(pxds.Files,1)));

for fold_idx = 1 : num_folds
    
    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,fold_idx);

    numTrainingImages = numel(imdsTrain.Files);
    numTestingImages = numel(imdsTest.Files);

    % data augmentation
    dsTrain = combine(imdsTrain, pxdsTrain);
    
    angle = [-45 45];
    xTrans = [-10 10];
    yTrans = [-10 10];
    dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data, angle, xTrans, yTrans));

    % U-Net model setting
    imageSize = [256, 256 1]; % input image size
    numClasses = numel(classes); % class setting
    
    % load U-Net architecture
    encoderDepth = 5;
    lgraph = unetLayers(imageSize, numClasses,'EncoderDepth',encoderDepth);
    
    % Classweight setting
%     tbl = countEachLabel(pxdsTrain);
%     
%     frequency = tbl.PixelCount/sum(tbl.PixelCount);
%     
%     imageFreq = sqrt(tbl.PixelCount ./ tbl.ImagePixelCount) .^1.5;
%     classWeights = median(imageFreq) ./ imageFreq;
%     
%     pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
%     lgraph = replaceLayer(lgraph,"Segmentation-Layer",pxLayer);
    
    
    % model option setting (best optimization hyperparameter)
    options = trainingOptions('adam', ...
        'InitialLearnRate', bayesianop(fold_idx).bestop{1,1}.InitialLearnRate, ...
        'MaxEpochs', 40, ...
        'L2Regularization', bayesianop(fold_idx).bestop{1,1}.L2Regularization, ...
        'GradientThreshold',bayesianop(fold_idx).bestop{1,1}.GradientThreshold, ...
        'MiniBatchSize', bayesianop(fold_idx).bestop{1,1}.miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'Plots','training-progress', ...
        'Verbose', 1, ...
        'ExecutionEnvironment','gpu');
    
    % U-Net model train
    [trainedNet, info] = trainNetwork(dsTrain,lgraph,options);
    
    % save the fold_idx U-Net train model to struct
    result_model(fold_idx).net{1,:} = trainedNet;
    
    % evaluate the k fold U-Net model
    path =  'D:\yachae_sw\CTImages\seg_data_100\';
    if ~exist([path,'Pred'], 'dir')
        mkdir([path,'Pred'])
    end
    tempdir = 'D:\yachae_sw\CTImages\seg_data_100\Pred\';
    pxdsResults = semanticseg(imdsTest, result_model(fold_idx).net{1},'MiniBatchSize',10,'WriteLocation',tempdir,'Verbose',false);
    
    metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
    MeanDice = average(generalizedDice(pxdsResults,pxdsTest));

    MeanIoU = metrics.DataSetMetrics.MeanIoU;

    result_model(fold_idx).IoU{1} = MeanIoU;
   
    % count the pixel(actual and predicted)
    test_count_end = test_count_st + size(pxdsTest.Files,1) - 1;

    for i = 1 : size(pxdsTest.Files, 1)

        pxpath = strjoin(pxdsTest.Files(i), '\n');
        image = imread(pxpath);

        [~, fileName, ~] = fileparts(pxpath);
        resultString = fileName(1:8);
        result_space(test_count_st + i - 1).Name{1} = resultString;

        px_space = 1;
        j = 1;
        for j = 1: size(result_ps,1)
            name2 = num2str(result_ps(j,1));
            if resultString == name2
                px_space = result_ps(j,2);
            end
        end

        countValueSAT = sum(image(:) == 255);
        result_space(test_count_st + i - 1).SAT{1} = int32(px_space * countValueSAT);

        countValueVAT = sum(image(:) == 170);
        result_space(test_count_st + i - 1).VAT{1} = int32(px_space * countValueVAT);

        countValueMuscle = sum(image(:) == 85);
        result_space(test_count_st + i - 1).Muscle{1} = int32(px_space * countValueMuscle);


        predPath = strjoin(pxdsResults.Files(i), '\n');
        predimage = imread(predPath);
        countSATpred = sum(predimage(:) == 1);
        result_space(test_count_st + i - 1).predSAT{1} = int32(px_space * countSATpred);

        countVATpred = sum(predimage(:) == 2);
        result_space(test_count_st + i - 1).predVAT{1} = int32(px_space * countVATpred);

        countMusclepred = sum(predimage(:) == 3);
        result_space(test_count_st + i - 1).predMuscle{1} = int32(px_space * countMusclepred);

    end

    %  reset
    if exist([path,'Pred'], 'dir')
        rmdir('D:\yachae_sw\CTImages\seg_data_100\Pred', 's');
    end
    
    clear trainedNet

    test_count_st = 1 + test_count_end;
end

% svae trained unet model
save('model_result.mat', 'result_model','-v7.3','-nocompression');
save('result_space.mat', 'result_space');
writetable(struct2table(result_space), 'result_space.xlsx')

%% best model evaluate

% max IoU Value and Index
maxValue = result_model(1).IoU{:};
maxIndex = 1;
for i = 2:length(result_model)
    currentValue = result_model(i).IoU{:};
    if currentValue > maxValue
        maxValue = currentValue;
        maxIndex = i;
    end
end
% calculate the best IoU

if ~exist([path,'Pred'], 'dir')
    mkdir([path,'Pred'])
end

[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,maxIndex);
pxdsResults = semanticseg(imdsTest, result_model(maxIndex).net{1,1},'MiniBatchSize',10,'WriteLocation',tempdir,'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
metrics.DataSetMetrics
metrics.ConfusionMatrix
data = metrics.DataSetMetrics;
save('DataSetMetrics.mat','data');
metrics.ClassMetrics
result_cm = metrics.ClassMetrics;
save('ClassMetrics.mat',"result_cm");


%% pred result overlay

Base_Overlay = 'D:\yachae_sw\CTImages\seg_data_100\Result\';

% result folder reset
if exist([Base_Overlay,'TestCT'], 'dir')
    rmdir('D:\yachae_sw\CTImages\seg_data_100\Result\TestCT', 's');
end

if exist([Base_Overlay,'TestLabel'], 'dir')
    rmdir('D:\yachae_sw\CTImages\seg_data_100\Result\TestLabel', 's');
end

if exist([Base_Overlay,'Overlay'], 'dir')
    rmdir('D:\yachae_sw\CTImages\seg_data_100\Result\Overlay', 's');
end  

if exist([Base_Overlay,'Labeloverlay'], 'dir')
    rmdir('D:\yachae_sw\CTImages\seg_data_100\Result\Labeloverlay', 's');
end 

if exist([Base_Overlay,'Predlabel'], 'dir')
    rmdir('D:\yachae_sw\CTImages\seg_data_100\Result\Predlabel', 's');
end

% make result folder
if ~exist([Base_Overlay,'TestCT'], 'dir')
    mkdir([Base_Overlay,'TestCT'])
end

if ~exist([Base_Overlay,'TestLabel'], 'dir')
    mkdir([Base_Overlay,'TestLabel'])
end

if ~exist([Base_Overlay,'Overlay'], 'dir')
    mkdir([Base_Overlay,'Overlay'])
end  

if ~exist([Base_Overlay,'Labeloverlay'], 'dir')
    mkdir([Base_Overlay,'Labeloverlay'])
end 

if ~exist([Base_Overlay,'Predlabel'], 'dir')
    mkdir([Base_Overlay,'Predlabel'])
end

for k = 1 : size(pxdsResults.Files,1)

    % test dicom image save
    I = readimage(imdsTest,k);
    Temp = [Base_Overlay 'TestCT\' 'CTimage_' num2str(k) '.png'];
    imwrite(I, Temp,BitDepth=8);

    % test label image save
    expectedResult = readimage(pxdsTest,k);
    actual = uint8(expectedResult);
    labelpng = actual .* 63;
    Temp1 = [Base_Overlay 'Testlabel\' 'Label_' num2str(k) '.png'];
    imwrite(labelpng, Temp1,BitDepth=8);

    % dicom image file and pred label overlay

    predlabel = semanticseg(I, result_model(maxIndex).net{1,1});
    overlay1 = labeloverlay(I,predlabel,'IncludedLabels',"Muscle",'Colormap','jet');
    Temp2 = [Base_Overlay 'Overlay\' 'overlay_' num2str(k) '.png'];
    imwrite(overlay1, Temp2,BitDepth=8);

    % label file and pred label overlay

    expected = uint8(predlabel);
    overlay2 = imshowpair(actual,expected);
    overlay2_matrix = getimage(overlay2);
    Temp3 = [Base_Overlay 'Labeloverlay\' 'labeloverlay_' num2str(k) '.png'];
    imwrite(overlay2_matrix, Temp3,BitDepth=8);

    % pred label to png

    predlabel = pxdsResults.Files{k,1};
    loadimage = imread(predlabel);
    predpng = loadimage .* 63;
    Temp4 = [Base_Overlay 'Predlabel\' 'predlabel_' num2str(k) '.png'];
    imwrite(predpng, Temp4,BitDepth=8);
end
%% pred result overlay

Base_Overlay = 'D:\yachae_sw\CTImages\segment_data\Result\';

for fold = 1 % : num_folds
    % result folder reset
    if exist([Base_Overlay,'TestCT'], 'dir')
        rmdir('D:\yachae_sw\CTImages\segment_data\Result\TestCT', 's');
    end
    
    if exist([Base_Overlay,'TestLabel'], 'dir')
        rmdir('D:\yachae_sw\CTImages\segment_data\Result\TestLabel', 's');
    end
    
    if exist([Base_Overlay,'OverlayM'], 'dir')
        rmdir('D:\yachae_sw\CTImages\segment_data\Result\OverlayM', 's');
    end 
    
    if exist([Base_Overlay,'OverlayM'], 'dir')
        rmdir('D:\yachae_sw\CTImages\segment_data\Result\OverlayV', 's');
    end 
    
    if exist([Base_Overlay,'OverlayM'], 'dir')
        rmdir('D:\yachae_sw\CTImages\segment_data\Result\OverlayS', 's');
    end 
    
    if exist([Base_Overlay,'Labeloverlay'], 'dir')
        rmdir('D:\yachae_sw\CTImages\segment_data\Result\Labeloverlay', 's');
    end 
    
    if exist([Base_Overlay,'Predlabel'], 'dir')
        rmdir('D:\yachae_sw\CTImages\segment_data\Result\Predlabel', 's');
    end
    
    % make result folder
    if ~exist([Base_Overlay,'TestCT'], 'dir')
        mkdir([Base_Overlay,'TestCT'])
    end
    
    if ~exist([Base_Overlay,'TestLabel'], 'dir')
        mkdir([Base_Overlay,'TestLabel'])
    end
    
    if ~exist([Base_Overlay,'OverlayM'], 'dir')
        mkdir([Base_Overlay,'OverlayM'])
    end  
    
    if ~exist([Base_Overlay,'OverlayV'], 'dir')
        mkdir([Base_Overlay,'OverlayV'])
    end  
    
    if ~exist([Base_Overlay,'OverlayS'], 'dir')
        mkdir([Base_Overlay,'OverlayS'])
    end  
    
    
    if ~exist([Base_Overlay,'Labeloverlay'], 'dir')
        mkdir([Base_Overlay,'Labeloverlay'])
    end 
    
    if ~exist([Base_Overlay,'Predlabel'], 'dir')
        mkdir([Base_Overlay,'Predlabel'])
    end
    
    for k = 1 : size(seg_result(fold).predictLabelTot,1)
    
        % test dicom image save
        Temp = [Base_Overlay 'TestCT\' 'CTimage_' num2str(k) '.png'];
        imwrite(seg_result(fold).testimage{k,1}, Temp,BitDepth=8);
    
        % test label image save
        Temp1 = [Base_Overlay 'Testlabel\' 'Label_' num2str(k) '.png'];
        imwrite(seg_result(fold).labelpng{k,1}, Temp1,BitDepth=8);
    
        % dicom image file and pred label overlay
        Temp2 = [Base_Overlay 'OverlayM\' 'overlayM_' num2str(k) '.png'];
        imwrite(seg_result(fold).overlayM{k,1}, Temp2,BitDepth=8);
    
        Temp2 = [Base_Overlay 'OverlayV\' 'overlayV_' num2str(k) '.png'];
        imwrite(seg_result(fold).overlayV{k,1}, Temp2,BitDepth=8);

        Temp2 = [Base_Overlay 'OverlayS\' 'overlayS_' num2str(k) '.png'];
        imwrite(seg_result(fold).overlayS{k,1}, Temp2,BitDepth=8);

        % label file and pred label overlay
    
        Temp3 = [Base_Overlay 'Labeloverlay\' 'labeloverlay_' num2str(k) '.png'];
        imwrite(seg_result(fold).overlayall_matrix{k,1}, Temp3,BitDepth=8);
    
        % pred label to png
    
        Temp4 = [Base_Overlay 'Predlabel\' 'predlabel_' num2str(k) '.png'];
        imwrite( seg_result(fold).predpng{k,1}, Temp4,BitDepth=8);
    end
end
%% dice scores

dice_scores = zeros(length(pxdsResults.Files), 4);

% calculate dice score
for i = 1:length(pxdsResults.Files)
    % actual data load
    image1 = readimage(pxdsTest, i);

    % pred data load
    image2 = readimage(pxdsResults, i);

    % calculate dice score
    dice_scores(i,:) = transpose(dice(image1, image2));
end

% result visualization
mean_score = mean(dice_scores);
disp(['Average Dice Score: ' num2str(mean_score)]);

boxplot(dice_scores, classes,'Colors','rb')
xlabel('class')
ylabel('Dice score')
title('prediction result dice scores')
