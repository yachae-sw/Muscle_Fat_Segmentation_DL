function [Final_Objective] = UNETCVOpti(optimVars, imds, pxds, fold)
    miniBatchSize = optimVars.miniBatchSize;
    MaxEpochs = 40;

    [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds,fold);
    % bar(1:numel(classes),frequency)
    % xticks(1:numel(classes)) 
    % xticklabels(tbl.Name)
    % xtickangle(45)
    % ylabel('Frequency')

    % data augmentation
    dsTrain = combine(imdsTrain, pxdsTrain);
    
    xTrans = [-10 10];
    yTrans = [-10 10];
    augdsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));

    options = trainingOptions('adam', ...
        'InitialLearnRate', optimVars.InitialLearnRate, ...
        'MaxEpochs', MaxEpochs, ...
        'GradientThreshold',optimVars.GradientThreshold, ...
        'GradientThresholdMethod',char(optimVars.GradientThresholdMethod),...
        'MiniBatchSize', miniBatchSize, ...
        'L2Regularization', optimVars.L2Regularization, ...
        'SequenceLength','longest', ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',10, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose', 0, ...
        'ExecutionEnvironment','gpu');


    imageSize = [256, 256 1];
    classes = ["VAT", "Muscle","SAT","background"];
    numClasses = numel(classes); % class setting
    
    % load U-Net architecture
    encoderDepth = 5;
    lgraph = unetLayers(imageSize, numClasses,'EncoderDepth',encoderDepth);
    
    % Classweight setting
    % tbl = countEachLabel(pxdsTrain);
    % pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',[1;optimVars.classWeightVAT;optimVars.classWeightMuscle;1]);
    % lgraph = replaceLayer(lgraph,"Segmentation-Layer",pxLayer);

    % evaluate the U-Net model
    Base_pred = 'D:\yachae_sw\CTImages\segment_data\';
    
    if exist([Base_pred,'Pred'], 'dir')
        rmdir('D:\yachae_sw\CTImages\segment_data\Pred', 's');
    end

    if ~exist([Base_pred,'Pred'], 'dir')
        mkdir([Base_pred,'Pred'])
    end  

    [trainedNet, info] = trainNetwork(augdsTrain,lgraph,options);
    seg_result(fold).trainedNet = trainedNet;
    seg_result(fold).accloss = info;

    tempdir = 'D:\yachae_sw\CTImages\segment_data\Pred\';
    pxdsResults = semanticseg(imdsTest, trainedNet,'MiniBatchSize',10,'WriteLocation',tempdir,'Verbose',false);
    % metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
    
    pxdsTest.ReadSize = size(pxdsTest.Files,1);
    pxdsResults.ReadSize = size(pxdsResults.Files,1);
    groundTruthLabelTot  = read(pxdsTest);
    % hasdata(pxdsTest)
    seg_result(fold).predictLabelTot  = read(pxdsResults);
    
    for i = 1 : size(seg_result(fold).predictLabelTot)
        diceResult(i,:) = zeros(size(classes));
        groundTruthLabel = groundTruthLabelTot{i};
        predictLabel = seg_result(fold).predictLabelTot{i};
        for j = 1:length(classes)
            diceResult(i,j) = dice(groundTruthLabel == classes(j), predictLabel == classes(j));
        end
    end

    metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
    seg_result(fold).DataSetMetrics = metrics.DataSetMetrics;
    seg_result(fold).ConfusionMatrix = metrics.ConfusionMatrix;
    seg_result(fold).ClassMetrics = metrics.ClassMetrics;

    seg_result(fold).eachdiceresult = diceResult;
    seg_result(fold).diceresult = mean(diceResult);
    MeanDice  =  mean(mean(diceResult));
    seg_result(fold).MeanDice = MeanDice;
    % MeanIoU = metrics.DataSetMetrics.MeanIoU;
    
    confusionMatrix = table2array(seg_result(fold).ConfusionMatrix);

    % TP1, TP2, TP3, TP4 값을 추출
    TP1 = confusionMatrix(1, 1);
    TP2 = confusionMatrix(2, 2);
    TP3 = confusionMatrix(3, 3);
    TP4 = confusionMatrix(4, 4);
    
    % 각 TP에 대한 Jaccard, 민감도, 특이도 계산
    seg_result(fold).jaccard(1, 1) = TP1 / (TP1 + sum(confusionMatrix(1, [2, 3, 4])) + sum(confusionMatrix([2, 3, 4], 1)));
    seg_result(fold).sensitivity(1, 1) = TP1 / (TP1 + sum(confusionMatrix(1, [2, 3, 4])));
    seg_result(fold).specificity(1, 1) = (sum(confusionMatrix([2, 3, 4], [2, 3, 4])) - sum(confusionMatrix([2, 3, 4], 1)) - sum(confusionMatrix(1, [2, 3, 4])) + TP1) / ...
        (sum(confusionMatrix([2, 3, 4], [2, 3, 4])) - sum(confusionMatrix([2, 3, 4], 1)) - sum(confusionMatrix(1, [2, 3, 4])) + TP1 + sum(confusionMatrix([1], [2, 3, 4])));
    
    seg_result(fold).jaccard(2, 1) = TP2 / (TP2 + sum(confusionMatrix(2, [1, 3, 4])) + sum(confusionMatrix([1, 3, 4], 2)));
    seg_result(fold).sensitivity(2, 1) = TP2 / (TP2 + sum(confusionMatrix(2, [1, 3, 4])));
    seg_result(fold).specificity(2, 1) = (sum(confusionMatrix([1, 3, 4], [1, 3, 4])) - sum(confusionMatrix([1, 3, 4], 2)) - sum(confusionMatrix(2, [1, 3, 4])) + TP2) / ...
        (sum(confusionMatrix([1, 3, 4], [1, 3, 4])) - sum(confusionMatrix([1, 3, 4], 2)) - sum(confusionMatrix(2, [1, 3, 4])) + TP2 + sum(confusionMatrix([2], [1, 3, 4])));
    
    seg_result(fold).jaccard(3, 1) = TP3 / (TP3 + sum(confusionMatrix(3, [1, 2, 4])) + sum(confusionMatrix([1, 2, 4], 3)));
    seg_result(fold).sensitivity(3, 1) = TP3 / (TP3 + sum(confusionMatrix(3, [1, 2, 4])));
    seg_result(fold).specificity(3, 1) = (sum(confusionMatrix([1, 2, 4], [1, 2, 4])) - sum(confusionMatrix([1, 2, 4], 3)) - sum(confusionMatrix(3, [1, 2, 4])) + TP3) / ...
        (sum(confusionMatrix([1, 2, 4], [1, 2, 4])) - sum(confusionMatrix([1, 2, 4], 3)) - sum(confusionMatrix(3, [1, 2, 4])) + TP3 + sum(confusionMatrix([3], [1, 2, 4])));
    
    seg_result(fold).jaccard(4, 1) = TP4 / (TP4 + sum(confusionMatrix(4, [1, 2, 3])) + sum(confusionMatrix([1, 2, 3], 4)));
    seg_result(fold).sensitivity(4, 1) = TP4 / (TP4 + sum(confusionMatrix(4, [1, 2, 3])));
    seg_result(fold).specificity(4, 1) = (sum(confusionMatrix([1, 2, 3], [1, 2, 3])) - sum(confusionMatrix([1, 2, 3], 4)) - sum(confusionMatrix(4, [1, 2, 3])) + TP4) / ...
        (sum(confusionMatrix([1, 2, 3], [1, 2, 3])) - sum(confusionMatrix([1, 2, 3], 4)) - sum(confusionMatrix(4, [1, 2, 3])) + TP4 + sum(confusionMatrix([4], [1, 2, 3])));

    seg_result(fold).jaccardresult = mean(seg_result(fold).jaccard);
    seg_result(fold).sensitivityresult = mean(seg_result(fold).sensitivity);
    seg_result(fold).specificityresult = mean(seg_result(fold).specificity);


    cnt3 = 1;
    for k = 1 : size(seg_result(fold).predictLabelTot,1)
        seg_result(fold).testimage{cnt3,1} = readimage(imdsTest,k);

        expectedResult = readimage(pxdsTest,k);
        actual = uint8(expectedResult);
        seg_result(fold).labelpng{cnt3,1} = (actual -1)  .* 85;  

        predlabel = pxdsResults.Files{k,1};
        loadimage = imread(predlabel);
        seg_result(fold).predpng{cnt3,1} = (loadimage -1).* 85;

        seg_result(fold).overlayM{cnt3,1} = labeloverlay(seg_result(fold).testimage{cnt3,1},seg_result(fold).predictLabelTot{k,1},'IncludedLabels',"Muscle",'Colormap','jet');
        seg_result(fold).overlayV{cnt3,1} = labeloverlay(seg_result(fold).testimage{cnt3,1},seg_result(fold).predictLabelTot{k,1},'IncludedLabels',"VAT",'Colormap','jet');
        seg_result(fold).overlayS{cnt3,1} = labeloverlay(seg_result(fold).testimage{cnt3,1},seg_result(fold).predictLabelTot{k,1},'IncludedLabels',"SAT",'Colormap','jet');

        expected = uint8(seg_result(fold).predictLabelTot{k,1});
        overlay2 = imshowpair(actual,expected);
        seg_result(fold).overlayall_matrix{cnt3,1} = getimage(overlay2);

        cnt3 = cnt3 + 1;
        clear actual predlabel loadimage expected overlay2
    end

    fprintf('jaccard = %d, dicescore = %d, sensitivity = %d, specificity = %d \n', seg_result(fold).jaccardresult, MeanDice, seg_result(fold).sensitivityresult, seg_result(fold).specificityresult);

    % Final_Objective = 1 - MeanIoU;
    Final_Objective = 1 - MeanDice;

    savefolder = "D:\yachae_sw\code\segmentation\bayesian_result";
    filename = num2str(fold) + "_" + num2str(Final_Objective) + ".mat";
    savefile = fullfile(savefolder,filename);

    save(savefile, 'seg_result');

    close all
end

