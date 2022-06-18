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


Row = 200;
Col = 88;
WindowRow = 25;
WindowCol = 25;
LineWidth = 2;
Scale = 3;

Diff_TGRS = mean(abs(I_TGRS-double(ImgRef)),3);
Diff_CT = mean(abs(I_CT-double(ImgRef)),3);
Diff_LGC = mean(abs(I_LGC-double(ImgRef)),3);

O_Ref = ShowEnlargedPatch(ImgRef, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_LRMS = ShowEnlargedPatch(I_LRMS, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_TGRS = ShowEnlargedPatch(I_TGRS, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_CT = ShowEnlargedPatch(I_CT, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_LGC = ShowEnlargedPatch(I_LGC, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_PXS = ShowEnlargedPatch(I_PXS, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_PCA = ShowEnlargedPatch(I_PCA, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_Indusion = ShowEnlargedPatch(I_Indusion , Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_IHS = ShowEnlargedPatch(I_IHS, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_GS = ShowEnlargedPatch(I_GS, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_Brovey = ShowEnlargedPatch(I_Brovey, Row, Col, WindowRow, WindowCol, LineWidth, Scale);
O_PNN = ShowEnlargedPatch(I_PNN, Row, Col, WindowRow, WindowCol, LineWidth, Scale);

imwrite(uint8(O_Ref),'O_Ref.png');
imwrite(uint8(O_LRMS),'O_LRMS.png');
imwrite(uint8(O_TGRS),'O_TGRS.png');
imwrite(uint8(O_CT),'O_CT.png');
imwrite(uint8(O_LGC),'O_LGC.png');
imwrite(uint8(O_PXS),'O_PXS.png');
imwrite(uint8(O_PCA),'O_PCA.png');
imwrite(uint8(O_Indusion),'O_Indusion.png');
imwrite(uint8(O_IHS),'O_IHS.png');
imwrite(uint8(O_GS),'O_GS.png');
imwrite(uint8(O_Brovey),'O_Brovey.png');
imwrite(uint8(O_PNN),'O_PNN.png');

% imwrite(uint8(O_LGC),'O_LGC.png');
% imwrite(uint8(O_LGC),'O_LGC.png');
% imwrite(uint8(O_LGC),'O_LGC.png');
% imwrite(uint8(O_LGC),'O_LGC.png');
% imwrite(uint8(O_LGC),'O_LGC.png');
% imwrite(uint8(O_LGC),'O_LGC.png');

%imshow(uint8(Diff_TGRS)),figure,imshow(uint8(Diff_CT)),figure,imshow(uint8(Diff_LGC));

%imshow(O_Ref),figure,imshow(uint8(O_TGRS)),figure,imshow(uint8(O_CT)),figure,imshow(uint8(O_LGC));