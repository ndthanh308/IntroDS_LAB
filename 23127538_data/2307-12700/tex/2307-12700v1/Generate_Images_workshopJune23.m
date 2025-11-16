figure
% load depth_SBR1.00_PPP10.0_DS4.0_v6.mat
load depth_SBR1.00_PPP1.0_DS4.0_v6.mat
subplot(2,2,1); imagesc( squeeze(MSd(1,1,:,:)));caxis([0.05 0.3])
title(['PPP=1, SBR=1'],'FontSize',16)
ylabel('Class.','FontSize',14)
subplot(2,2,3); imagesc( squeeze(depth(4,1,:,:)));caxis([0.05 0.3])
ylabel('BU3D (x4)','FontSize',14)

% load depth_SBR10.00_PPP10.0_DS4.0_v6.mat
load depth_SBR1.00_PPP10.0_DS4.0_v6.mat

subplot(2,2,2); imagesc( squeeze(MSd(1,1,:,:)));caxis([0.05 0.3])
title(['PPP=10, SBR=1'],'FontSize',16)
ylabel('Class.','FontSize',14)
subplot(2,2,4); imagesc( squeeze(depth(4,1,:,:)));caxis([0.05 0.3])
ylabel('BU3D (x4)','FontSize',14)

 
saveName = ('Art_SRx4');%('Bowling_RMSE_Ref_3Algos_ExpoBack1_Bayesian_CorrVar');
saveas(gcf, saveName, 'fig');
print Art_SRx4.png -dpng -r512