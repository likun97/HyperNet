clc
clear all

ImgGF = imread('GF1_Ref_50.tif');
ImgRef = ImgGF(:,:,[1,2,3]);
ImgGF = load('I_LRMS_50.mat');
I_LRMS = ImgGF.I_MS(:,:,[3,2,1]);
ImgGF = load('I_TGRS_50.mat');
I_TGRS = ImgGF.I_TGRS(:,:,[3,2,1]);
ImgGF = load('I_PXS_50.mat');
I_PXS = ImgGF.I_PXS(:,:,[3,2,1]);
ImgGF = load('I_PCA_50.mat');
I_PCA = ImgGF.I_PCA(:,:,[3,2,1]);
ImgGF = load('I_LGC_50.mat');
I_LGC = ImgGF.I_LGC(:,:,[3,2,1]);
ImgGF = load('I_Indusion_50.mat');
I_Indusion = ImgGF.I_Indusion(:,:,[3,2,1]);
ImgGF = load('I_IHS_50.mat');
I_IHS = ImgGF.I_IHS(:,:,[3,2,1]);
ImgGF = load('I_GS_50.mat');
I_GS = ImgGF.I_GS(:,:,[3,2,1]);
ImgGF = load('I_Brovey_50.mat');
I_Brovey = ImgGF.I_Brovey(:,:,[3,2,1]);
ImgGF = load('I_CT_50.mat');
I_CT = ImgGF.I_CT(:,:,[3,2,1]);
ImgGF = load('I_PNN_50.mat');
I_PNN = ImgGF.fusion(:,:,[3,2,1]);

Diff_TGRS = uint8(mean(abs(I_TGRS-double(ImgRef)),3));
Diff_CT = uint8(mean(abs(I_CT-double(ImgRef)),3));
Diff_LGC = uint8(mean(abs(I_LGC-double(ImgRef)),3));

Diff_PXS = uint8(mean(abs(I_PXS-double(ImgRef)),3));
Diff_PCA = uint8(mean(abs(I_PCA-double(ImgRef)),3));
Diff_Indusion = uint8(mean(abs(I_Indusion-double(ImgRef)),3));
Diff_IHS = uint8(mean(abs(I_IHS-double(ImgRef)),3));
Diff_GS = uint8(mean(abs(I_GS-double(ImgRef)),3));
Diff_Brovey = uint8(mean(abs(I_Brovey-double(ImgRef)),3));
Diff_PNN = uint8(mean(abs(I_PNN-double(ImgRef)),3));

MaxR = 128;

imshow(Diff_PNN);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_PNN_50.png');

figure
imshow(Diff_TGRS);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_TGRS_50.png');

figure
imshow(Diff_CT);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_CT_50.png');

figure
imshow(Diff_LGC);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_LGC_50.png');

figure
imshow(Diff_PXS);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_PXS_50.png');

figure
imshow(Diff_PCA);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_PCA_50.png');

figure
imshow(Diff_Indusion);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_Indusion_50.png');

figure
imshow(Diff_IHS);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_IHS_50.png');

figure
imshow(Diff_GS);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_GS_50.png');

figure
imshow(Diff_Brovey);
caxis([0,MaxR]);
colormap(jet); 
F1=getframe;
imwrite(uint8(F1.cdata), 'Diff_Brovey_50.png');


%F=getframe(gcf);




