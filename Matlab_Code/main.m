clc;close all;clear;
load ("Dicom3.mat");
load("bestParameterR.mat");
% load("pixelspace.mat");

%% load dataset(Image and Mask)

datasetpath = 'E:\Medical Image processing\AP\segmentation\segmentation\seg_data_100\';
imageDir = fullfile(datasetpath,'image');
maskDir = fullfile(datasetpath,'mask');


imds = imageDatastore(imageDir);

% check image
% I = readimage(imds,51);
% imshow(I)

classes = ["SAT", "VAT", "Muscle","background"];

labelIDs   = [255 170 85 000];

pxds = pixelLabelDatastore(maskDir,classes,labelIDs);

num_folds = 5;
test_count_st = 1;
test_count_end = 1;

result_model = struct('net',cell(1,num_folds));
result_space = struct('Name', cell(1, size(pxds.Files,1)));
num_folds = 5;

for fold_idx = 1 : num_folds

    %% design optimization variable
    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,fold_idx);
    MaxEpochs = 30;



    
    dsTrain = combine(imdsTrain, pxdsTrain);
    
    angle = [-45 45];
    xTrans = [-10 10];
    yTrans = [-10 10];
    dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data, angle, xTrans, yTrans));
    imageSize = [256, 256 1];
    classes = ["SAT", "VAT", "Muscle","background"];
    numClasses = numel(classes); % class setting
    

    options = trainingOptions('adam', ...
        'InitialLearnRate', bayesianop(fold_idx).bestop{1}.InitialLearnRate, ...
        'MaxEpochs', MaxEpochs, ...
        'GradientThreshold',bayesianop(fold_idx).bestop{1}.GradientThreshold, ...
        'MiniBatchSize', bayesianop(fold_idx).bestop{1}.miniBatchSize, ...
        'L2Regularization', bayesianop(fold_idx).bestop{1}.L2Regularization, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 1, ...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3), ...
        'ExecutionEnvironment','gpu');

    % load U-Net architecture
    encoderDepth = 5;
    lgraph = unetLayers(imageSize, numClasses,'EncoderDepth',encoderDepth);

   
    % evaluate the U-Net model
    Base_pred = 'E:\Medical Image processing\AP\newdataanaly\';

    if exist([Base_pred,'Pred'], 'dir')
        rmdir('E:\Medical Image processing\AP\newdataanaly\Pred','s');
    end

    if ~exist([Base_pred,'Pred'], 'dir')
        mkdir([Base_pred,'Pred'])
    end

    result_model(fold_idx).net = trainNetwork(dsTrain,lgraph,options);
    tempdir = 'E:\Medical Image processing\AP\newdataanaly\Pred';
    pxdsResults = semanticseg(imdsTest, result_model(fold_idx).net,'MiniBatchSize',10,'WriteLocation',tempdir,'Verbose',false);
    

    metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);

    MeanIoU = metrics.DataSetMetrics.MeanIoU;

    result_model(fold_idx).IoU{1} = MeanIoU;
   


end
save('result_model','result_model','-v7.3')

%% postprocessing
clear; close; clc;
load('result_model')
load('Housfieldunit.mat')
load('Pixelspacing.mat')
datasetpath = 'E:\Medical Image processing\AP\segmentation\segmentation\seg_data_100\';
Base_Overlay = 'E:\Medical Image processing\AP\newdataanaly\';
imageDir = fullfile(datasetpath,'image');
maskDir = fullfile(datasetpath,'mask');
imds = imageDatastore(imageDir);
classes = ["SAT", "VAT", "Muscle","background"];
labelIDs   = [255 170 85 000];
pxds = pixelLabelDatastore(maskDir,classes,labelIDs);
num_folds = 5;
tempdir = 'E:\Medical Image processing\AP\newdataanaly\Pred\';
postpreddir = 'E:\Medical Image processing\AP\newdataanaly\PredPost\';

% 
% % result folder reset
% if exist([Base_Overlay,'TestCT'], 'dir')
%     rmdir('E:\Medical Image processing\AP\newdataanaly\TestCT', 's');
% end
% 
% if exist([Base_Overlay,'TestLabel'], 'dir')
%     rmdir('E:\Medical Image processing\AP\newdataanaly\TestLabel', 's');
% end
% 
% if exist([Base_Overlay,'Overlay'], 'dir')
%     rmdir('E:\Medical Image processing\AP\newdataanaly\Overlay', 's');
% end  
% 
% if exist([Base_Overlay,'Labeloverlay'], 'dir')
%     rmdir('E:\Medical Image processing\AP\newdataanaly\Labeloverlay', 's');
% end 
% 
% if exist([Base_Overlay,'Predlabel'], 'dir')
%     rmdir('E:\Medical Image processing\AP\newdataanaly\Predlabel', 's');
% end
% 
% % make result folder
% if ~exist([Base_Overlay,'TestCT'], 'dir')
%     mkdir([Base_Overlay,'TestCT'])
% end
% 
% if ~exist([Base_Overlay,'TestLabel'], 'dir')
%     mkdir([Base_Overlay,'TestLabel'])
% end
% 
% if ~exist([Base_Overlay,'Overlay'], 'dir')
%     mkdir([Base_Overlay,'Overlay'])
% end  
% 
% if ~exist([Base_Overlay,'Labeloverlay'], 'dir')
%     mkdir([Base_Overlay,'Labeloverlay'])
% end 
% 
% if ~exist([Base_Overlay,'Predlabel'], 'dir')
%     mkdir([Base_Overlay,'Predlabel'])
% end


for fold_idx = 1 : num_folds
    clear tst_Pixelspacing
   
    testIdx = fold_idx:5:length(Housfieldunit);
    trainingIdx = setdiff(1:length(Housfieldunit),testIdx);
    tst_Housfieldunit = Housfieldunit(testIdx);
    tst_Pixelspacing = Pixelspacing(testIdx);
    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,fold_idx);
    
    pxdsResults = semanticseg(imdsTest, result_model(fold_idx).net,'MiniBatchSize',10,'WriteLocation',tempdir,'Verbose',false);
    

    metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);

    MeanIoU = metrics.DataSetMetrics.MeanIoU;

    result_model(fold_idx).preIoU = MeanIoU;
    result_model(fold_idx).result = confusion.getValues(table2array(metrics.ConfusionMatrix));
    [temp,referenceresult]=multiclass_metrics_special(table2array(metrics.ConfusionMatrix));
    result_model(fold_idx).classspecificity = referenceresult.Specificity;
    result_model(fold_idx).classsensitivity = referenceresult.Recall;

    for k = 1 : size(pxdsResults.Files)
        pdpath = strjoin(pxdsResults.Files(k), '\n');
        pred_image = imread(pdpath);
        re_housfieldSAT = round(imresize(tst_Housfieldunit(k).L3HounsSAT,[256, 256]));
        re_housfieldVAT = round(imresize(tst_Housfieldunit(k).L3HounsVAT,[256, 256]));        
        re_housfieldSM = round(imresize(tst_Housfieldunit(k).L3HounsSM,[256, 256])); 

        for i = 1 : size(pred_image,1)
            for j = 1 : size(pred_image,2)
                if (pred_image(i,j) == 1) && (re_housfieldSAT(i,j) == 1)
                    pred_image_post(i,j) = 255/255;
                    pred_image_post_cat(i,j) = 1;
                elseif(pred_image(i,j) == 2) && (re_housfieldVAT(i,j) == 1)
                    pred_image_post(i,j) = 170/255;   
                    pred_image_post_cat(i,j) = 2;
                elseif (pred_image(i,j) == 3) && (re_housfieldSM(i,j) == 1)
                    pred_image_post(i,j) = 85/255;
                    pred_image_post_cat(i,j) = 3;
                else
                    pred_image_post(i,j) = 0;
                    pred_image_post_cat(i,j) = 0;
                end
            end
        end
        Temp = string(postpreddir)+string(fold_idx)+'\'+string(tst_Housfieldunit(k).L3JPGImageName);
        imwrite(pred_image_post, Temp,BitDepth=8)
        

        % test dicom image save
        I = readimage(imdsTest,k);
        Temp = [Base_Overlay 'TestCT\'  num2str(fold_idx) '\' 'CTimage_' num2str(k) '.png'];
        imwrite(I, Temp,BitDepth=8);

         % test label image save
         expectedResult = readimage(pxdsTest,k);
         actual = uint8(expectedResult);
         labelpng = actual .* 63;
         Temp1 = [Base_Overlay 'Testlabel\'  num2str(fold_idx) '\' 'Label_' num2str(k) '.png'];
         imwrite(labelpng, Temp1,BitDepth=8);


         % dicom image file and pred label overlay
         predlabel =  categorical(pred_image_post*255,labelIDs,classes);
         overlay1 = labeloverlay(I,predlabel,'IncludedLabels',["Muscle" "SAT", "VAT"], 'Colormap','jet');
         Temp2 = [Base_Overlay  'Overlay\'   num2str(fold_idx) '\' 'overlay_' num2str(k) '.png'];
         imwrite(overlay1, Temp2,BitDepth=8);

         % label file and pred label overlay
         predlabel = pxdsResults.Files{k,1};
         expected = uint8(predlabel);
         overlay2 = imshowpair(actual,expected);
         overlay2_matrix = getimage(overlay2);
         Temp3 = [Base_Overlay 'Labeloverlay\' num2str(fold_idx) '\' 'labeloverlay_'  num2str(k) '.png'];
         imwrite(overlay2_matrix, Temp3,BitDepth=8);

         % pred label to png


         loadimage = imread(predlabel);
         predpng = loadimage .* 63;
         Temp4 = [Base_Overlay 'Predlabel\' num2str(fold_idx) '\' 'predlabel_'  num2str(k) '.png'];
         imwrite(predpng, Temp4,BitDepth=8);         

    end

    clear preimds
    predimds = pixelLabelDatastore(string(postpreddir)+string(fold_idx)+'\',classes,labelIDs);


    metrics = evaluateSemanticSegmentation(predimds,pxdsTest,'Verbose',false);
    MeanIoU = metrics.DataSetMetrics.MeanIoU;
    result_model(fold_idx).classIoU = metrics.ClassMetrics;
    result_model(fold_idx).IoU = MeanIoU;
    clear dice_scores
    for k = 1 : size(pxdsResults.Files)

        img1 = readimage(predimds,k);
        img2 = readimage(pxdsTest,k);
        dice_scores(k,:) = transpose(dice(img1,img2));

    end
    result_model(fold_idx).dice_scores_all = dice_scores;
    result_model(fold_idx).dice_scores = mean(dice_scores);
    result_model(fold_idx).mean_dice_score = mean(mean(dice_scores));
    result_model(fold_idx).sensitivity = result_model(fold_idx).result.Sensitivity;
    result_model(fold_idx).Specificity = result_model(fold_idx).result.Specificity;


     % count the pixel(actual and predicted)
   

     for i = 1 : size(pxdsTest.Files, 1)


         pxpath = strjoin(pxdsTest.Files(i), '\n');
         image = imread(pxpath);

         countValueSAT = sum(image(:) == 255);
         result_model(fold_idx).result_space(i,1) = int32(tst_Pixelspacing(i) * countValueSAT);

         countValueVAT = sum(image(:) == 170);
         result_model(fold_idx).result_space(i,2) = int32(tst_Pixelspacing(i) * countValueVAT);

         countValueMuscle = sum(image(:) == 85);
         result_model(fold_idx).result_space(i,3) = int32(tst_Pixelspacing(i) * countValueMuscle);

         predPath = strjoin(predimds.Files(i), '\n');
         predimage = imread(predPath);

         countSATpred = sum(predimage(:) == 255);
         result_model(fold_idx).result_space(i,4)= int32(tst_Pixelspacing(i) * countSATpred);

         countVATpred = sum(predimage(:) == 170);
         result_model(fold_idx).result_space(i,5) = int32(tst_Pixelspacing(i) * countVATpred);

         countMusclepred = sum(predimage(:) == 85);
         result_model(fold_idx).result_space(i,6) = int32(tst_Pixelspacing(i) * countMusclepred);


        

     end


end

save('result_model.mat','result_model','-v7.3')
predpost_image = imread(Temp);





for fold_idx = 1:num_folds
    
    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,fold_idx);

    numTrainingImages = numel(imdsTrain.Files);
    numTestingImages = numel(imdsTest.Files);

    % data augmentation
    dsTrain = combine(imdsTrain, pxdsTrain);
    
    angle = [-45 45];
    xTrans = [-10 10];
    yTrans = [-10 10];
 
    
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

    MeanIoU = metrics.DataSetMetrics.MeanIoU;

    result_model(fold_idx).IoU{1} = MeanIoU;
   
   

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

Base_Overlay = 'E:\Medical Image processing\AP\newdataanaly\';

% result folder reset
if exist([Base_Overlay,'TestCT'], 'dir')
    rmdir('E:\Medical Image processing\AP\newdataanaly\TestCT', 's');
end

if exist([Base_Overlay,'TestLabel'], 'dir')
    rmdir('E:\Medical Image processing\AP\newdataanaly\TestLabel', 's');
end

if exist([Base_Overlay,'Overlay'], 'dir')
    rmdir('E:\Medical Image processing\AP\newdataanaly\Overlay', 's');
end  

if exist([Base_Overlay,'Labeloverlay'], 'dir')
    rmdir('E:\Medical Image processing\AP\newdataanaly\Labeloverlay', 's');
end 

if exist([Base_Overlay,'Predlabel'], 'dir')
    rmdir('E:\Medical Image processing\AP\newdataanaly\Predlabel', 's');
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
