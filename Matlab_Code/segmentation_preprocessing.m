clc;close all;clear;
%% Double Dicom data

% load dicom file path
Base1 = 'D:\yachae_sw\CTImages\';  % adjust to your needs
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
[AP_Data_Label] = readcell(fullfile(Base1, 'CT_Mask_nii_100/CT_100_label_20230807.xlsx'));

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

cell_max = size(SubFolder,2);
Dicom2 = struct('Dicomname', cell(1, cell_max), 'Subject', cell(1, cell_max));

count = 0;
for j = 1 : size(SubFolder, 2)
    str1 = dir(fullfile(Source_dir1{j}, '*.dcm*'));
    for k = AP_Data_Label{j+1, 3} : AP_Data_Label{j+1, 10}
        if AP_Data_Label{j+1, 3} == 1
            try
                Temp = [Source_dir1{j}, str1(k).name];
                Dicom2(j).Dicomname{k,:} = [str1(k).name];
                dicomimage = dicomread(Temp);
                dicomimage32 = int32(dicomimage);
                info = dicominfo(Temp);
 
                % image resize
                reimage = imresize(dicomimage32, [256 256],"nearest");

                for b = 1 : size(reimage,1)
                    for c = 1 :size(reimage,2)
                        hounsfieldImage(b,c) = int32(reimage(b,c))*info.RescaleSlope + int32(info.RescaleIntercept); % control HU
                        % Hu max value setting
                        if hounsfieldImage(b,c) > 200
                            hounsfieldImage(b,c) = 200;
                        % Hu min value setting
                        elseif hounsfieldImage(b,c) < -200
                            hounsfieldImage(b,c) = -200;
                        end
                    end
                end

                Dicom2(j).Subject{k,:} = hounsfieldImage;
            catch err
                break;
            end
        else
            % start file number is not 1
            minus = AP_Data_Label{j+1, 3}-1;
            try
                Temp = [Source_dir1{j}, str1(k).name];
                Dicom2(j).Dicomname{k-minus,:} = [str1(k).name];
                dicomimage = dicomread(Temp);
                dicomimage32 = int32(dicomimage);
                info = dicominfo(Temp);
                reimage = imresize(dicomimage32, [256 256],"nearest");
                for b = 1 : size(reimage,1)
                    for c = 1 :size(reimage,2)
                        hounsfieldImage(b,c) = int32(reimage(b,c))*info.RescaleSlope + int32(info.RescaleIntercept);
                        if hounsfieldImage(b,c) > 200
                            hounsfieldImage(b,c) = 200;
                        elseif hounsfieldImage(b,c) < -200
                            hounsfieldImage(b,c) = -200;
                        end
                    end
                end
                Dicom2(j).Subject{k-minus,:} = hounsfieldImage;
            catch err
                break;
            end
        end
    end
    count = count + 1;
    disp(count)
end

%% Loading nii form file and extracting muscle(label = 1)

Base2 = 'D:\yachae_sw\CTImages\CT_Mask_nii_100\';  % adjust to your needs
List2 = dir(fullfile(Base2, '*.nii*'));
SubFolder2 = {List2.name};
Source_dir2 = cellfun(@(c)[Base2 c ],SubFolder2,'uni',false);

count = 0;
for j = 1 : size(Dicom2,2)
    str1 = Source_dir2{j};
    Temp1 = [str1(1:strfind(str1,'.')-1),'.nii'];
    V = niftiread(Temp1);
    L = length(unique(V));
    for k = 1 : size(V,3)
        if L == 1 
            Vr = imresize(V(:,:,k),[256 256]);
        else
            % zero for the rest
            for ii = 1 : size(V,1)
                for jj = 1 : size(V,2)
                    if  V(ii,jj,k) == 4
                        V(ii,jj,k) = 3;
                    elseif V(ii,jj,k) == 1
                        V(ii,jj,k) = 2;
                    elseif V(ii,jj,k) == 5
                        V(ii,jj,k) = 1;
                    else
                        V(ii,jj,k) = 0;
                    end
                end
            end
            Vr = imresize(V(:,:,k),[256 256],'nearest');
        %     for iii = 1 : size(Vr,1)
        %         for jjj = 1 : size(Vr,2)
        %             if Vr(iii,jjj) == 0.5
        %                 Vrr(iii,jjj) = 1;
        %             elseif Vr(iii,jjj) == 2
        %                 Vrr(iii,jjj) = 4;
        %             elseif Vr(iii,jjj) == 2.5
        %                 Vrr(iii,jjj) = 4;
        %             elseif Vr(iii,jjj) == 3
        %                 Vrr(iii,jjj) = 6;
        %             elseif Vr(iii,jjj) == 3.5
        %                 Vrr(iii,jjj) = 6;
        %             elseif Vr(iii,jjj) == 5
        %                 Vrr(iii,jjj) = 6;
        %             else
        %                 Vrr(iii,jjj) = Vr(iii,jjj);
        %             end
        %         end
        %     end
        %     for iiii = 1 : size(Vrr,1)
        %         for jjjj = 1 : size(Vrr,2)
        %             if  Vrr(iiii,jjjj) == 1
        %                 Vrrr(iiii,jjjj) = 1;
        %             elseif Vrr(iiii,jjjj) == 4
        %                 Vrrr(iiii,jjjj) = 2;
        %             elseif Vrr(iiii,jjjj) == 6
        %                 Vrrr(iiii,jjjj) = 3;
        %             else
        %                 Vrrr(iiii,jjjj) = 0;
        %             end
        %         end
        %     end
        end
        Dicom2(j).SegLabel{size(V,3)-k+1,:} = Vr;
    end
    %         Newdicom1{j,:} = [Source_dir2{:,i}, str_trim1{j,:},'_1','.dcm']
    %         dicomwrite(Olddicom1{j,:}, Newdicom1{j,:});
    count = count + 1;
    disp(count)
end

clear Vr;

%% Label contrast and mask angle change

i=1;
j=1;
cnt = 1;
NumClass = 2;
ImageSize = 256;
for i = 1 : size(Dicom2,2)
    cnt = 1;
    for j = 1 : AP_Data_Label{i+1,5}
%         Dicom(i).RawImage{cnt,:} = Dicom(i).Subject{j,:};
        if AP_L3_Label{i,:}(j,1) == 0 % L3 label
            Dicom2(i).SegImage{cnt,:} = zeros(ImageSize,ImageSize);
            Dicom2(i).Label{cnt,:} = AP_L3_Label{i,:}(j,1);

            str = Dicom2(i).Dicomname{j,:};
            str_trim = str(1:strfind(str,'.')-1);
            new_name = replace(str_trim,"'","");
            Dicom2(i).JPGImageName{cnt,:} = [new_name,'.png'];
        else
%             Dicom(i).SegImage{cnt,:} = ones(size(Dicom(i).Subject{j,:},1),size(Dicom(i).Subject{j,:},2));
            Dicom2(i).SegImage{cnt,:} = imadjust(mat2gray(fliplr(rot90(Dicom2(i).SegLabel{j,:},3)),[0 3]),[0 1]);
            Dicom2(i).Label{cnt,:} = AP_L3_Label{i,:}(j,1);
            
            str = Dicom2(i).Dicomname{j,:};
            str_trim = str(1:strfind(str,'.')-1);
            new_name = replace(str_trim,"'","");
            Dicom2(i).JPGImageName{cnt,:} = [new_name,'.png'];
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
for i = 1 : size(Dicom2,2)
    for j = 1 : size(Dicom2(i).SegImage,1)
        Dicom2(i).RawImageRev{cnt1,:} = int16(Dicom2(i).Subject{j,:});
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

count = 0;
for i = 1 : size(Dicom2,2)
    cnt = 1;
    cnt3 = 1;
    for j = 1 : size(Dicom2(i).RawImageRev,1)
        cnt1 = 1;
        cnt2 = 1;
        loadimage = Dicom2(i).RawImageRev{j,:};
        for b = 1 : 256
            for c = 1 : 256
                if loadimage(b,c) <= -200
                    cnt1 = cnt1 + 1;
                elseif loadimage(b,c) >= 200
                    cnt2 = cnt2 + 1;
                end
            end            
        end
        stretchlimst = round(cnt1 / (256 * 256),4);
        stretchlimend = 1 - max(0.02, round(cnt2 / (256 * 256),4));
        Dicom2(i).stretchlim{j,1} = stretchlimst;
        Dicom2(i).stretchlim{j,2} = stretchlimend;

        Dicom2(i).ImageAdjust{cnt,:} = imadjust(Dicom2(i).RawImageRev{j,:},stretchlim(Dicom2(i).RawImageRev{j,:},[stretchlimst stretchlimend]),[]);
        cnt = cnt + 1;
    end
    for j = 1 : size(Dicom2(i).RawImageRev,1)
        if AP_L3_Label{i,:}(j,1) == 0
            continue;
        else
            Temp1_1 = Dicom2(i).ImageAdjust{j,:};
            Temp2 = Dicom2(i).SegImage{j,:};
            
%             Dicom(i).L3AddImage{cnt,:} = imfuse(Temp1,Temp2,'montage');
            Dicom2(i).L3AddImageResize{cnt3,:} = imfuse(Temp1_1,Temp2,'montage');
%             Dicom(i).L3OverlayImageResize{cnt,:} = labeloverlay(Temp1_1,Temp2_2);
%             Dicom(i).L3AddImageResize1{cnt,:} = imfuse(Dicom(i).OverlayImageResize{j,:},Temp2_1,'montage');
            Dicom2(i).L3JPGImageName{cnt3,:} = Dicom2(i).JPGImageName{j,:};
            cnt3 = cnt3 + 1;
        end
    end
    count = count + 1;
    disp(count)
end
% i=4;
% j=4;
% figure()
% imshow(Dicom2(i).L3AddImageResize{j,:})

%% image and label combine to png

AP = struct;
WriteOrNot = 1;

Base3 = 'D:\yachae_sw\CTImages\All_CT_Combine\';  % adjust to your needs
i=1;
j=1;
for i = 144 % pre_start_num : pre_end_num
    for j = 1 : size(Dicom2(i).L3JPGImageName,1)
        AP(i).L3JPGImageName{j,:} = Dicom2(i).L3JPGImageName{j,:};
        if WriteOrNot == 1
            Temp = [Base3,'L3seg\',Dicom2(i).L3JPGImageName{j,:}];
            
            if ~exist([Base3,'L3seg'], 'dir')
                mkdir([Base3,'L3seg'])
            end    
            
            AP(i).L3JPGWriteImage{j,:} = (imadjust(Dicom2(i).L3AddImageResize{j,:}));
            imwrite(Dicom2(i).L3AddImageResize{j,:}, Temp,BitDepth=16);
        else
            continue;
        end

    end
end
% figure()
% imshow(AP(i).L3JPGWriteImage{j,:})

%% segmentation data save

Base4 = 'D:\yachae_sw\CTImages\segment_data\';  % adjust to your needs
i=1;
j=1;
WriteOrNot = 1;
for i = 1 : size(Dicom2,2)
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
for i = 1 : size(Dicom2,2)
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

save("D:\yachae_sw\code\segmentation\Dicomseg.mat",'Dicom2','-mat','-v7.3')
% save([Base4,'AP.mat'],'AP','-v2','-nocompression');