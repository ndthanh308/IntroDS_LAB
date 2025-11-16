function [X10norm, X20norm, mu10, mu20, std10, std20] = fuseUpNorm(X10, X20, shift)
 
    mu20 = mean(X20,[1 2]);
    ms20 = mean(X20.^2,[1 2]);
    var20 = ms20 - mu20.^2;
    std20 = var20.^0.5;
    X20norm = shift + (X20-mu20)./std20;

    % LPF (MTF gains) im10
    ratio = 2;
    h = genMTF_v3(ratio,'',1);
    n10 = size(X10,3);
    X10lp = X10;
    for n = 1:n10
        X10lp(:,:,n) = imfilter(X10(:,:,n),h(:,:));
    end
    
    % optional
    %X10lp = imresize(X10lp,1/ratio);
    % 

    mu10 = mean(X10lp,[1 2]);
    ms10 = mean(X10lp.^2,[1 2]);
    var10 = ms10 - mu10.^2;
    std10 = var10.^0.5;
    X10norm = shift + (X10-mu10)./std10;
end

