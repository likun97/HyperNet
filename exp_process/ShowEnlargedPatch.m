function O_img = ShowEnlargedPatch(Img, Row, Col, WindowRow, WindowCol, LineWidth, Scale)

%  Row, Col:  感兴趣区域的左上角横纵坐标（在图像表现为行、列）
%%%%%%%%%%%%%%%%%%要求Row, Col均大于线宽
% WindowRow, WindowCol: 感兴趣区域大小
% LineWidth:  线的宽度
% Scale:  感兴趣区域放大的倍数
% 函数输出：
% O_img: 合成后的输出图像

[height,width,~] = size(Img);
Patch(:,:,:) = Img(Row:Row-1+WindowRow, Col:Col-1+WindowCol,:);
Interpolation_Method = 'nearest';
Enlarged = imresize(Patch, Scale, Interpolation_Method);

[m, n,~] = size(Enlarged);

% EnlargedShowStartRow = height - LineWidth-m+1;     %%显示在左下 
% EnlargedShowStartCol = 1 + LineWidth;

EnlargedShowStartRow =1 + LineWidth ;
EnlargedShowStartCol = width - LineWidth-n+1;      %%显示在右上 
 
% EnlargedShowStartRow =1 + LineWidth ;
% EnlargedShowStartCol = 1 + LineWidth;            %%显示在左上

EnlargedShowStartRow = height - LineWidth-m+1; 
EnlargedShowStartCol = width - LineWidth-n+1;    %%显示在右下 
 

Img(EnlargedShowStartRow:EnlargedShowStartRow+m-1,EnlargedShowStartCol:EnlargedShowStartCol + n - 1,:) = Enlarged(:,:,:);

Row = Row-1;Col=Col-1;WindowCol = WindowCol+1;WindowRow = WindowRow+1;

O_img = drawRect(Img,[Col Row],[WindowCol,WindowRow],2 );
O_img = drawRect(O_img,[EnlargedShowStartCol-1 EnlargedShowStartRow-1],[n+1,m+1],2 );



