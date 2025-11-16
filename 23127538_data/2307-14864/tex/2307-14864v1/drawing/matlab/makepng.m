%load ./10m/S2_10.mat
%load ./20m/S2_20.mat

load ./Patches/S2_10_NORM.mat
load ./Patches/S2_20_NORM.mat
Pred = getfield(load('./Patches/Predict_Proposed_Finetune_FullRef_NORM.mat'),'im20');

[X10norm, X20norm, mu10, mu20, std10, std20] = fuseUpNorm(im10, im20, 2);
Pred_norm = 2 + (Pred-mu20)./std20;

X10rgb = uint8(60*X10norm(:,:,[3 2 1]));
X20rgb = uint8(60*X20norm(:,:,[5 3 1]));
Pred_rgb = uint8(60*Pred_norm(:,:,[5 3 1]));

figure(1);
ax(1) = subplot(1,3,1); imshow(X10rgb,[]);
ax(2) = subplot(1,3,2); imshow(imresize(X20rgb,2,'nearest'),[]);
ax(3) = subplot(1,3,3); imshow(Pred_rgb,[]);
linkaxes(ax,'xy');



%%%% Selezione zoom nelle coordinate a bassa risoluzione
%i = 10; j = 13; d = 80; 
i = 174; j = 6; d = 60; 
%%%%


I = 2*i-1; J = 2*j-1; D = 2*d;

X10 = X10rgb(I:I+D-1,J:J+D-1,:);
X20 = X20rgb(i:i+d-1,j:j+d-1,:);
pred = Pred_rgb(I:I+D-1,J:J+D-1,:);

figure(2);
s(1) = subplot(1,3,1); imshow(X10,[]);
s(2) = subplot(1,3,2); imshow(imresize(X20,2,'nearest'),[]);
s(3) = subplot(1,3,3); imshow(pred,[]);
linkaxes(s,'xy');

%%% Eventuale salvataggio in PNG
%imwrite(X10,'X10.png');
%imwrite(X20,'X20.png');
%imwrite(pred,'pred.png');