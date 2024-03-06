clc;close all;clear;
%% Double Dicom data

pre_start_num = 1;
pre_end_num = 100;

% load dicom file path
Base1 = 'E:\Medical Image processing\AP\Dataset\DICOM\';  % adjust to your needs
List = dir(fullfile(Base1, 'CT_DCM_100', '*.*'));
List = List([List.isdir]);
SubFolder = {List.name};
SubFolder(ismember(SubFolder, {'.', '..'})) = [];
Source_dir1 = cellfun(@(c)[Base1 'CT_DCM_100\' c '\'],SubFolder,'uni',false);

% make each folder
% for i = 1 : length(Source_dir1)
%     A = SubFolder{:,i};
%     B = 'DICOM';
%     mkdir (A,B)
% end

% load excel file
[AP_Data_Label] = readcell(fullfile('E:\Medical Image processing\AP\CT_100_label_20230807.xlsx'));

% load to L3 location
AP_L3_Label = cell(size(AP_Data_Label, 1) - 1, 1);
for i = 2 : size(AP_Data_Label, 1)
    L3_Answer_Start = AP_Data_Label{i, 6};
    L3_Answer_End = AP_Data_Label{i, 7};
    L3_End_Reverse=fliplr(AP_Data_Label{i,5}-L3_Answer_Start)+1;
    L3_Start_Reverse=fliplr(AP_Data_Label{i,5}-L3_Answer_End)+1;
    AP_L3_Label{i-1} = zeros(AP_Data_Label{i, 5}, 1);
    for j = 1 : AP_Data_Label{i, 5}
        if L3_Start_Reverse <= L3_End_Reverse
            AP_L3_Label{i-1}(L3_Start_Reverse:L3_End_Reverse) = 1;
        end
    end
end

%% Save name and data in struct form
count = 0;
cnt = 1;
cell_max = size(SubFolder,2);
Dicom3 = struct('Dicomname', cell(1, cell_max), 'Subject', cell(1, cell_max));
for j = pre_start_num : pre_end_num % size(SubFolder, 2)
% for j = 10
    str1 = dir(fullfile(Source_dir1{j}, '*.dcm*'));
    for k = AP_Data_Label{j+1, 3} : AP_Data_Label{j+1, 10}
         Temp = [Source_dir1{j}, str1(k).name];
            dicomimage = dicomread(Temp);
            dicomimage32 = int32(dicomimage);
            info = dicominfo(Temp);
        if k == 1


            pixeltemp = info.PixelSpacing(1);
        end

        if AP_Data_Label{j+1, 3} == 1

            Temp = [Source_dir1{j}, str1(k).name];
            Dicom3(j).Dicomname{k,:} = [str1(k).name];
            dicomimage = dicomread(Temp);
            dicomimage32 = int32(dicomimage);
            info = dicominfo(Temp);
            Dicom3(j).pixelspace{k,:} = pixeltemp(1);
            for b = 1 : size(dicomimage32,1)
                for c = 1 :size(dicomimage32,2)
                    hounsfieldImage(b,c) = int32(dicomimage32(b,c))*info.RescaleSlope + int32(info.RescaleIntercept);
                    % Hu max min setting

                    if (hounsfieldImage(b,c) <= 150) && (hounsfieldImage(b,c) >= -29)
                        hounsfieldImageMuscle(b,c) = 1;
                    else
                        hounsfieldImageMuscle(b,c) = 0;
                    end

                    if (hounsfieldImage(b,c) <= -50) && (hounsfieldImage(b,c) >= -150)
                        hounsfieldImageVAT(b,c) = 1;
                    else
                        hounsfieldImageVAT(b,c) = 0;
                    end

                    if (hounsfieldImage(b,c) <= 30) && (hounsfieldImage(b,c) >= -190)
                        hounsfieldImageSAT(b,c) = 1;
                    else
                        hounsfieldImageSAT(b,c) = 0;
                    end

                    if hounsfieldImage(b,c) > 1000
                        hounsfieldImage(b,c) = 1000;
                    elseif hounsfieldImage(b,c) < -200
                        hounsfieldImage(b,c) = -1023;
                    end
                end
            end
            Dicom3(j).HounsSM{k,:} = hounsfieldImageMuscle;
            Dicom3(j).HounsVAT{k,:} = hounsfieldImageVAT;
            Dicom3(j).HounsSAT{k,:} = hounsfieldImageSAT;
            Dicom3(j).Subject{k,:} = imresize(hounsfieldImage, [256 256]);
            % catch err
            % break;
        else
            minus = AP_Data_Label{j+1, 3}-1;
            Temp = [Source_dir1{j}, str1(k).name];
            Dicom3(j).Dicomname{k-minus,:} = [str1(k).name];
            dicomimage = dicomread(Temp);
            dicomimage32 = int32(dicomimage);
            info = dicominfo(Temp);
            Dicom3(j).pixelspace{k-minus,:} = pixeltemp(1);

            for b = 1 : size(dicomimage32,1)
                for c = 1 :size(dicomimage32,2)
                    hounsfieldImage(b,c) = int32(dicomimage32(b,c))*info.RescaleSlope + int32(info.RescaleIntercept);
                if (hounsfieldImage(b,c) <= 150) && (hounsfieldImage(b,c) >= -29)
                    hounsfieldImageMuscle(b,c) = 1;
                else
                    hounsfieldImageMuscle(b,c) = 0;
                end

                if (hounsfieldImage(b,c) <= -50) && (hounsfieldImage(b,c) >= -150)
                    hounsfieldImageVAT(b,c) = 1;
                else
                    hounsfieldImageVAT(b,c) = 0;
                end

                if (hounsfieldImage(b,c) <= 30) && (hounsfieldImage(b,c) >= -190)
                    hounsfieldImageSAT(b,c) = 1;
                else
                    hounsfieldImageSAT(b,c) = 0;
                end

                    if hounsfieldImage(b,c) > 1000
                        hounsfieldImage(b,c) = 1000;
                    elseif hounsfieldImage(b,c) < -200
                        hounsfieldImage(b,c) = -1023;
                    end
                end
            end

            Dicom3(j).HounsSM{k-minus,:} = hounsfieldImageMuscle;
            Dicom3(j).HounsVAT{k-minus,:} = hounsfieldImageVAT;
            Dicom3(j).HounsSAT{k-minus,:} = hounsfieldImageSAT;
            %    % Hu max min setting
            Dicom3(j).Subject{k-minus,:} = imresize(hounsfieldImage, [256 256]);
        end
        cnt = cnt +1;
    end
    count = count + 1;
    disp(count)
end
save("Dicom3","Dicom3",'-v7.3')

%% Loading nii form file and extracting muscle(label = 1, 2, 3)

Base2 = 'E:\Medical Image processing\AP\Dataset\DICOM\CT_Mask_nii_100\';  % adjust to your needs
List2 = dir(fullfile(Base2, '*.nii*'));
SubFolder2 = {List2.name};
Source_dir2 = cellfun(@(c)[Base2 c ],SubFolder2,'uni',false);

pre_start_num = 1;
pre_end_num = 100;

for j = pre_start_num : pre_end_num
    str1 = Source_dir2{j};
    Temp1 = [str1(1:strfind(str1,'.')-1),'.nii'];
    V = niftiread(Temp1);
    L = length(unique(V));
    for k = 1 : size(V,3)
        if L == 1 
            Vr = imresize(V(:,:,k),[256 256],"nearest");
        else
            % zero for the rest
            for ii = 1 : size(V,1)
                for jj = 1 : size(V,2)
                    if  V(ii,jj,k) == 1 
                        V(ii,jj,k) = 1;
                    elseif V(ii,jj,k) == 4
                        V(ii,jj,k) = 2;
                    elseif V(ii,jj,k) == 5
                        V(ii,jj,k) = 3;
                    else
                        V(ii,jj,k) = 0;
                    end
                end
            end
            Vr = imresize(V(:,:,k),[256 256],"nearest");
        end
        Dicom3(j).SegLabel{size(V,3)-k+1,:} = Vr;
    end
    %         Newdicom1{j,:} = [Source_dir2{:,i}, str_trim1{j,:},'_1','.dcm']
    %         dicomwrite(Olddicom1{j,:}, Newdicom1{j,:});
end

clear Vr;

%% Label contrast and mask angle change

i=1;
j=1;
cnt = 1;
NumClass = 2;
ImageSize = 256;
for i = pre_start_num : pre_end_num % size(Dicom2,2)
    cnt = 1;
    for j = 1 : AP_Data_Label{i+1,5}
%         Dicom(i).RawImage{cnt,:} = Dicom(i).Subject{j,:};
        if AP_L3_Label{i,:}(j,1) == 0 % L3 label
            Dicom3(i).SegImage{cnt,:} = zeros(ImageSize,ImageSize);
            Dicom3(i).Label{cnt,:} = AP_L3_Label{i,:}(j,1);

            str = Dicom3(i).Dicomname{j,:};
            str_trim = str(1:strfind(str,'.')-1);
            Dicom3(i).JPGImageName{cnt,:} = [str_trim,'.png'];
        else
%             Dicom(i).SegImage{cnt,:} = ones(size(Dicom(i).Subject{j,:},1),size(Dicom(i).Subject{j,:},2));
            Dicom3(i).SegImage{cnt,:} = imadjust(mat2gray(fliplr(rot90(Dicom3(i).SegLabel{j,:},3)),[0 3]),[0 1]);
            Dicom3(i).Label{cnt,:} = AP_L3_Label{i,:}(j,1);
            
            str = Dicom3(i).Dicomname{j,:};
            str_trim = str(1:strfind(str,'.')-1);
            Dicom3(i).JPGImageName{cnt,:} = [str_trim,'.png'];
        end
        cnt = cnt + 1;
    end
end


% figure()
% imshow(rot90(Dicom(i).SegImage{j,:}))
% figure()
% imshow(fliplr(rot90(Dicom(i).SegImage{j,:})))

%% imcontrast

cnt1 = 1;
for i = pre_start_num : pre_end_num % size(Dicom2,2)
    for j = 1 : size(Dicom3(i).SegImage,1)
        Dicom3(i).RawImageRev{cnt1,:} = int16(Dicom3(i).Subject{j,:});
        cnt1 = cnt1 + 1;
    end
    cnt1 = 1;
end
% Dicom2 = rmfield(Dicom2, 'Subject');

% example check
% close all
% i = 1;
% j = 39;
% figure()
% imshow(mat2gray(imadjust(uint16(Dicom2(i).RawImageRev{j,:}),stretchlim(Dicom2(i).RawImageRev{j,:}),[])),[])
% imshow(imadjust(uint16(Dicom(i).RawImageRev{j,:}),stretchlim(Dicom(i).RawImageRev{j,:}),[]))
% imcontrast
% figure()
% imshow(Dicom(i).SegImage{j,:})
% imshow(imadjust(fliplr(rot90(Dicom(i).SegLabel{j,:},3))),[])
% imshow(imadjust(mat2gray(fliplr(rot90(Dicom2(i).SegLabel{j,:},3)),[0 1]),[0 1]))
% imshow(imadjust(fliplr(rot90(Dicom(i).SegLabel{j,:}))),[])
% unique(fliplr(rot90(Dicom(i).SegLabel{j,:})))
% imcontrast


%% image and label imfuse to struct
Housfieldunit = struct;



cnt1 = 1;
for i = pre_start_num : pre_end_num
    cnt = 1;
    for j = 1 : size(Dicom3(i).RawImageRev,1)
        if AP_L3_Label{i,:}(j,1) == 0
            continue;
        else
            Temp1 = imadjust(Dicom3(i).RawImageRev{j,:},stretchlim(Dicom3(i).RawImageRev{j,:},[0.2 0.98]),[]); % imadjust(Dicom(i).RawImage{j,:},stretchlim(Dicom(i).RawImage{j,:}),[]);
            Temp2 = Dicom3(i).SegImage{j,:};
            for k = 1 : size(Temp1,1)
                for l = 1 : size(Temp1,2)
                    if Temp1(k,l) < 0
                        Temp1(k,l) = 0;
                    else
                        Temp1(k,l) = Temp1(k,l);
                    end
                end
            end
            Temp1_1 = imresize(Temp1,[256, 256]);
            Temp2_1 = imresize(Temp2,[256, 256]);
            for ii = 1 : size(Temp2_1,1)
                for jj = 1 : size(Temp2_1,2)
                    if Temp2_1(ii,jj) <1
                        Temp2_2(ii,jj) = 0;
                    elseif Temp2_1(ii,jj) == 1
                        Temp2_2(ii,jj) = 1;
                    elseif Temp2_1(ii,jj) == 2
                        Temp2_2(ii,jj) = 2;
                    elseif Temp2_1(ii,jj) == 3
                        Temp2_2(ii,jj) = 3;
                    else
                        Temp2_2(ii,jj) = 0;
                    end
                    %                     if Temp2_1(ii,jj) <= 0.1
                    %                         Temp2_2(ii,jj) = 0;
                    %                     elseif Temp2_1(ii,jj) > 0.1 && Temp2_1(ii,jj) < 1
                    %                         Temp2_2(ii,jj) = 1;
                    %                     elseif Temp2_1(ii,jj) >= 1
                    %                         Temp2_2(ii,jj) = 1;
                    %                     end
                end
            end
            %             Dicom(i).L3AddImage{cnt,:} = imfuse(Temp1,Temp2,'montage');
            Dicom3(i).L3AddImageResize{cnt,:} = imfuse(Temp1_1,Temp2_1,'montage');
            %             Dicom(i).L3OverlayImageResize{cnt,:} = labeloverlay(Temp1_1,Temp2_2);
            %             Dicom(i).L3AddImageResize1{cnt,:} = imfuse(Dicom(i).OverlayImageResize{j,:},Temp2_1,'montage');
            Dicom3(i).L3JPGImageName{cnt,:} = Dicom3(i).JPGImageName{j,:};
            Dicom3(i).L3HounsSM{cnt,:} = Dicom3(i).HounsSM{j,:};
            Dicom3(i).L3HounsVAT{cnt,:} = Dicom3(i).HounsVAT{j,:};
            Dicom3(i).L3HounsSAT{cnt,:} = Dicom3(i).HounsSAT{j,:};
            Dicom3(i).L3pixelspace{cnt,:} = Dicom3(i).pixelspace{j,:};
            cnt = cnt + 1;


        end

    end
    for j = 1 : size(Dicom3(i).L3AddImageResize,1)
        Housfieldunit(cnt1).L3HounsSM = Dicom3(i).L3HounsSM{j,:};
        Housfieldunit(cnt1).L3HounsVAT = Dicom3(i).L3HounsVAT{j,:};
        Housfieldunit(cnt1).L3HounsSAT = Dicom3(i).L3HounsSAT{j,:};
        Housfieldunit(cnt1).L3JPGImageName = Dicom3(i).L3JPGImageName{j,:};
        Pixelspacing(cnt1) = Dicom3(i).L3pixelspace{j,:};
        cnt1 = cnt1 + 1;
    end
end
temp = 0;
for i = 1: size(Dicom3,2)
    temp = temp + size(Dicom3(i).L3HounsSAT,1);
end
% i=4;
% j=4;
% figure()
% imshow(Dicom2(i).L3AddImageResize{j,:})
% save("Dicom3","Dicom3",'-v7.3')
save("Housfieldunit","Housfieldunit",'-v7.3')
save("Pixelspacing","Pixelspacing")
%% image and label combine to png

AP = struct;
WriteOrNot = 1;
cnt = 1;
Base3 = 'E:\Medical Image processing\AP\segmentation\segmentation\';  % adjust to your needs
i=1;
j=1;
for i = pre_start_num : pre_end_num
    for j = 1 : size(Dicom3(i).L3JPGImageName,1)
        AP(i).L3JPGImageName{j,:} = Dicom3(i).L3JPGImageName{j,:};
        if WriteOrNot == 1
            Temp = [Base3,'L3_seg_100\',Dicom3(i).L3JPGImageName{j,:}];
            
            if ~exist([Base3,'L3_seg_100'], 'dir')
                mkdir([Base3,'L3_seg_100'])
            end    
            
            AP(i).L3JPGWriteImage{j,:} = (imadjust(Dicom3(i).L3AddImageResize{j,:}));
            imwrite(Dicom3(i).L3AddImageResize{j,:}, Temp,BitDepth=16);
        else
            continue;
        end
    end
end
% figure()
% imshow(AP(i).L3JPGWriteImage{j,:})

%% segmentation data save

Base4 = 'E:\Medical Image processing\AP\segmentation\segmentation\seg_data_100\';  % adjust to your needs
i=1;
j=1;
WriteOrNot = 1;
for i = pre_start_num : pre_end_num % size(Dicom2,2)
    for j = 1 : size(Dicom2(i).L3JPGImageName,1)
        AP(i).L3JPGImageName{j,:} = Dicom2(i).L3JPGImageName{j,:};
        if WriteOrNot == 1
            Temp = [Base4,'Image\',Dicom2(i).L3JPGImageName{j,:}];
            
            if ~exist([Base4,'Image'], 'dir')
                mkdir([Base4,'Image'])
            end    
            
            AP(i).L3Image{j,:} = (imadjust(Dicom2(i).L3AddImageResize{j,:}(:,1:256)));
            imwrite(Dicom2(i).L3AddImageResize{j,:}(:,1:256), Temp,BitDepth=8);
        else
            continue;
        end

    end
end

i=1;
j=1;
WriteOrNot = 1;
for i = pre_start_num : pre_end_num % size(Dicom2,2)
    for j = 1 : size(Dicom2(i).L3JPGImageName,1)
        AP(i).L3JPGImageName{j,:} = Dicom2(i).L3JPGImageName{j,:};
        if WriteOrNot == 1
            Temp = [Base4,'Mask\',Dicom2(i).L3JPGImageName{j,:}];
            
            if ~exist([Base4,'Mask'], 'dir')
                mkdir([Base4,'Mask'])
            end    
            
            AP(i).L3Mask{j,:} = (imadjust(Dicom2(i).L3AddImageResize{j,:}(:,257:512)));
            imwrite(Dicom2(i).L3AddImageResize{j,:}(:,257:512), Temp,BitDepth=8);
        else
            continue;
        end

    end
end

