function [Final_Objective] = UNETCVweightOpti(optimVars, imds, pxds, fold)
    miniBatchSize = optimVars.miniBatchSize;
    MaxEpochs = 30;

    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,fold);
    % bar(1:numel(classes),frequency)
    % xticks(1:numel(classes)) 
    % xticklabels(tbl.Name)
    % xtickangle(45)
    % ylabel('Frequency')

    % data augmentation
    dsTrain = combine(imdsTrain, pxdsTrain);
    
    angle = [-45 45];
    xTrans = [-10 10];
    yTrans = [-10 10];
    dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data, angle, xTrans, yTrans));

    options = trainingOptions('adam', ...
        'InitialLearnRate', optimVars.InitialLearnRate, ...
        'MaxEpochs', MaxEpochs, ...
        'GradientThreshold',optimVars.GradientThreshold, ...
        'MiniBatchSize', miniBatchSize, ...
        'L2Regularization', optimVars.L2Regularization, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 1, ...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3), ...
        'ExecutionEnvironment','gpu');


    imageSize = [256, 256 1];
    classes = ["SAT", "VAT", "Muscle","background"];
    numClasses = numel(classes); % class setting
    
    % load U-Net architecture
    encoderDepth = 5;
    lgraph = unetLayers(imageSize, numClasses,'EncoderDepth',encoderDepth);
    
    % Classweight setting
    % imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
    % classWeights = median(imageFreq) ./ imageFreq;
    
    pxLayer = pixelClassificationLayer('Name','labels','Classes',classes,'ClassWeights',...
        [optimVars.SATweight, optimVars.VATweight, optimVars.Muscleweight,1]);
    lgraph = replaceLayer(lgraph,"Segmentation-Layer",pxLayer);
    
    % evaluate the U-Net model
    Base_pred = 'D:\yachae_sw\CTImages\seg_data_100\';
    
    if exist([Base_pred,'Pred'], 'dir')
        rmdir('D:\yachae_sw\CTImages\seg_data_100\Pred', 's');
    end

    if ~exist([Base_pred,'Pred'], 'dir')
        mkdir([Base_pred,'Pred'])
    end  

    trainedNet = trainNetwork(dsTrain,lgraph,options);
    tempdir = 'D:\yachae_sw\CTImages\seg_data_100\Pred\';
    pxdsResults = semanticseg(imdsTest, trainedNet,'MiniBatchSize',10,'WriteLocation',tempdir,'Verbose',false);
    
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
    mean_score = mean(mean(dice_scores));




    metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
    
    MeanIoU = metrics.DataSetMetrics.MeanIoU;

    Final_Objective = 1 - mean_score;
    fileName = num2str(Final_Objective) + ".mat";
            save(fileName,'trainedNet','Final_Objective','options','lgraph')

    close all
end


