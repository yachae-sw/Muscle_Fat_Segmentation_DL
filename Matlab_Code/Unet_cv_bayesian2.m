clc;close all;clear;

%% load dataset(Image and Mask)

datasetpath = 'D:\Medical Image processing\AP\segmentation\segmentation\seg_data_100\';
imageDir = fullfile(datasetpath,'image');
maskDir = fullfile(datasetpath,'mask');

imds = imageDatastore(imageDir);

% check image
% I = readimage(imds,51);
% imshow(I)

classes = ["SAT", "VAT", "Muscle","background"];

labelIDs   = [255 170 85 000];

pxds = pixelLabelDatastore(maskDir,classes,labelIDs);
 

% check muscle iamge
% C = readimage(pxds,51);
% B = labeloverlay(I,C,'IncludedLabels',"Muscle");
% imshow(B)



%% bayesian optimization

num_folds = 5;
bayesianop = struct('bestop', cell(1, num_folds));
for i = 1 : num_folds

    %% design optimization variable
    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,i);

    for  j = 1 : size(pxdsTrain.Files,1)
        temp1 = 0;
        temp2 = 0;
        temp3 = 0;
        temp4 = 0;
        temp =    imread(pxdsTrain.Files{j});
        for k = 1 : size(temp,1)

            for kk = 1 : size(temp,2)
                if temp(k,kk) == 255
                    temp1 = temp1 + 1 ;
                elseif temp(k,kk) == 170
                    temp2 = temp2 + 1;
                elseif temp(k,kk) ==85
                    temp3 = temp3 + 1 ;
                elseif temp(k,kk) == 0
                    temp4 = temp4 + 1 ;
                end
                SATweightnum{i}(j) = temp1;
                VATweightnum{i}(j) = temp2;
                Muscleweightnum{i}(j) = temp3;
                backgroundweightnum{i}(j) = temp4;

            end
        end
    end
    
    maxminSATweight{i}(1) = max(SATweightnum{i});
    maxminSATweight{i}(2) = min(SATweightnum{i});
    maxminSATweight{i}(3) = mean(SATweightnum{i});

    maxminVATweight{i}(1) = max(VATweightnum{i});
    maxminVATweight{i}(2) = min(VATweightnum{i});
    maxminVATweight{i}(3) = mean(VATweightnum{i});

    if maxminVATweight{i}(2) == 0
        maxminVATweight{i}(2) =1;
    end
    maxminMuscleweight{i}(1) = max(Muscleweightnum{i});
    maxminMuscleweight{i}(2) = min(Muscleweightnum{i});
    maxminMuscleweight{i}(3) = mean(Muscleweightnum{i});
    if maxminMuscleweight{i}(2) == 0
        maxminMuscleweight{i}(2) =1;
    end
    maxminbackgroundweight{i}(1) = max(backgroundweightnum{i});
    maxminbackgroundweight{i}(2) = min(backgroundweightnum{i});
    maxminbackgroundweight{i}(3) = mean(backgroundweightnum{i});

    numpixel{i} = maxminbackgroundweight{i}(3) + maxminMuscleweight{i}(3) + maxminVATweight{i}(3) + maxminSATweight{i}(3);

    ratio{i}(3) = maxminMuscleweight{i}(3)/numpixel{i};
    ratio{i}(2) =  maxminVATweight{i}(3)/numpixel{i};
    ratio{i}(1) = maxminSATweight{i}(3)/numpixel{i};
    ratio{i}(4) = maxminbackgroundweight{i}(3)/numpixel{i};

    ratio_nom{i}(1) = ratio{i}(1)/ratio{i}(4);
    ratio_nom{i}(2) =  ratio{i}(2)/ratio{i}(4);
    ratio_nom{i}(3) =ratio{i}(3)/ratio{i}(4);
    ratio_nom{i}(4) =ratio{i}(4)/ratio{i}(4);

    optimVars = [
        optimizableVariable('L2Regularization',[0.00001 0.01],'Type','real')
        optimizableVariable('InitialLearnRate',[0.00001 0.001],'Type','real')
        optimizableVariable('miniBatchSize',[10 32],'Type','integer')
        optimizableVariable('GradientThreshold',[1 6],'Type','integer')
        optimizableVariable('SATweight',[0.5 20],'Type','real')
        optimizableVariable('VATweight',[0.5 20],'Type','real')
        optimizableVariable('Muscleweight',[0.5 20],'Type','real')
        ];


    % bayesian object function
    ObjFcn = @(optimVars) UNETCVweightOpti(optimVars, imds, pxds, i);

    BayesObjectR = bayesopt(ObjFcn,optimVars,...
        'AcquisitionFunctionName','expected-improvement-plus','PlotFcn',[],...
        'IsObjectiveDeterministic',false,...
        'MaxObjectiveEvaluations',50,...
        'UseParallel',false);

    % 'MaxObjectiveEvaluations',1,...

    % best parameter extraction
    bayesianop(i).bestop{1,:} = bestPoint(BayesObjectR);
end

save('bestParameterR.mat', 'bayesianop','BayesObjectR');


%% U-net model and 5-fold cross validation

test_count_st = 1;
test_count_end = 1;
load('pixelspace.mat');
result_model = struct('net',cell(1,num_folds));
result_space = struct('Name', cell(1, size(pxds.Files,1)));

for fold_idx = 1:num_folds
    
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
    tempdir = 'D:\Medical Image processing\AP\segmentation\segmentation\control_code\pred\';
    pxdsResults = semanticseg(imdsTest, result_model(fold_idx).net{1},'MiniBatchSize',10,'WriteLocation',tempdir,'Verbose',false);
    
    metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);

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
        result_space(test_count_st + i - 1).SAT = int32(px_space * countValueSAT);

        countValueVAT = sum(image(:) == 170);
        result_space(test_count_st + i - 1).VAT = int32(px_space * countValueVAT);

        countValueMuscle = sum(image(:) == 85);
        result_space(test_count_st + i - 1).Muscle = int32(px_space * countValueMuscle);


        predPath = strjoin(pxdsResults.Files(i), '\n');
        predimage = imread(predPath);
        countSATpred = sum(predimage(:) == 1);
        result_space(test_count_st + i - 1).predSAT = int32(px_space * countSATpred);

        countVATpred = sum(predimage(:) == 2);
        result_space(test_count_st + i - 1).predVAT = int32(px_space * countVATpred);

        countMusclepred = sum(predimage(:) == 3);
        result_space(test_count_st + i - 1).predMuscle = int32(px_space * countMusclepred);

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

Base_Overlay = 'D:\Medical Image processing\AP\segmentation\segmentation\control_code\';

% result folder reset
if exist([Base_Overlay,'TestCT'], 'dir')
    rmdir('D:\Medical Image processing\AP\segmentation\segmentation\control_code\TestCT', 's');
end

if exist([Base_Overlay,'TestLabel'], 'dir')
    rmdir('D:\Medical Image processing\AP\segmentation\segmentation\control_code\TestLabel', 's');
end

if exist([Base_Overlay,'Overlay'], 'dir')
    rmdir('D:\Medical Image processing\AP\segmentation\segmentation\control_code\TestLabel\Overlay', 's');
end  

if exist([Base_Overlay,'Labeloverlay'], 'dir')
    rmdir('D:\Medical Image processing\AP\segmentation\segmentation\control_code\TestLabel\Labeloverlay', 's');
end 

if exist([Base_Overlay,'Predlabel'], 'dir')
    rmdir('D:\Medical Image processing\AP\segmentation\segmentation\control_code\TestLabel\Predlabel', 's');
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
    overlay1 = labeloverlay(I,predlabel,'IncludedLabels',["SAT", "VAT", "Muscle"],'Colormap','jet');
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
