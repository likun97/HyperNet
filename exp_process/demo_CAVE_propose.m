  clear;clc;close all;
% addpath(genpath('.'));
%% Eveluation data ^_^
 
imgPath_Propose = 'A dir of .MAT data consisted of your fusion result and reference\';  
imgDir_Propose  = dir([imgPath_Propose '*.mat']);%遍历所有mat文件
Matrix0         = zeros(10,8);
sf              = 32; 

 
for i=1:length(imgDir_Propose)
s = imgDir_Propose(i).name;
s = s(1:end-4);
load([imgPath_Propose imgDir_Propose(i).name]);
S = (ref);
Z = (fusion);

% imwrite((S(:,:,[10 20 30])),[save_path s '_ref.png']);
% imwrite((Z(:,:,[10 20 30])),[save_path s '_Propose.png']);
[psnr4,rmse4, ergas4, sam4, uiqi4,ssim4,DD4,CC4] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1.0/sf);
Matrix0(i,:) = [psnr4,rmse4, ergas4, sam4, uiqi4,ssim4,DD4,CC4];

end

%%  imshow

img = imread('xxx.png'); 
img = img(1:380,101:480,:); 
LineWidth = 2;   Scale     = 4; 
Col       = 265; Row       = 80; 
WindowCol = 90;  WindowRow = 20;  
O_Img = ShowEnlargedPatch(img , Row, Col, WindowRow, WindowCol, LineWidth, Scale); 
figure,imshow(O_Img), title ('O_Img');
imwrite(O_Img, [save_path 'O_Img.png']);