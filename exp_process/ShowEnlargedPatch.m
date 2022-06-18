function O_img = ShowEnlargedPatch(Img, Row, Col, WindowRow, WindowCol, LineWidth, Scale)

%  Row, Col:  ����Ȥ��������ϽǺ������꣨��ͼ�����Ϊ�С��У�
%%%%%%%%%%%%%%%%%%Ҫ��Row, Col�������߿�
% WindowRow, WindowCol: ����Ȥ�����С
% LineWidth:  �ߵĿ��
% Scale:  ����Ȥ����Ŵ�ı���
% ���������
% O_img: �ϳɺ�����ͼ��

[height,width,~] = size(Img);
Patch(:,:,:) = Img(Row:Row-1+WindowRow, Col:Col-1+WindowCol,:);
Interpolation_Method = 'nearest';
Enlarged = imresize(Patch, Scale, Interpolation_Method);

[m, n,~] = size(Enlarged);

% EnlargedShowStartRow = height - LineWidth-m+1;     %%��ʾ������ 
% EnlargedShowStartCol = 1 + LineWidth;

EnlargedShowStartRow =1 + LineWidth ;
EnlargedShowStartCol = width - LineWidth-n+1;      %%��ʾ������ 
 
% EnlargedShowStartRow =1 + LineWidth ;
% EnlargedShowStartCol = 1 + LineWidth;            %%��ʾ������

EnlargedShowStartRow = height - LineWidth-m+1; 
EnlargedShowStartCol = width - LineWidth-n+1;    %%��ʾ������ 
 

Img(EnlargedShowStartRow:EnlargedShowStartRow+m-1,EnlargedShowStartCol:EnlargedShowStartCol + n - 1,:) = Enlarged(:,:,:);

Row = Row-1;Col=Col-1;WindowCol = WindowCol+1;WindowRow = WindowRow+1;

O_img = drawRect(Img,[Col Row],[WindowCol,WindowRow],2 );
O_img = drawRect(O_img,[EnlargedShowStartCol-1 EnlargedShowStartRow-1],[n+1,m+1],2 );



