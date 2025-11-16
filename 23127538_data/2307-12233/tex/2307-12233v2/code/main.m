% Numerical simulations for Cavallino project
% Marco Fabris 24/07/2022

close all
clear all
clc


% style specs and options
epx_ids = [1 2];
ss.graphshow = 1;
ss.lw = 2;
ss.ftsz = 30;
ss.ftszleg = 20;
ss.in = 'latex';
ss.fpos = [600 200 600 600];
ss.mygreen = [0 128 0]/255;
global step 
step = 1;

% adjacency matrices of the given topologies
m = 22;
A1 = [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
      0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
      0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
      0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
      0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0];

x01 = [7 2 2 5 10 6 3 8 3 6 7 9 10 6 2 2 3 9 3 9 3 10 4 2 3]';
trend1 = mean(x01);
x01 = x01-trend1;

% A2 = A1;
% %done = 0;
% t = 0;
% while t < 10 %&& ~done
% %     Eo2 = full(incidence(graph(A2)));
% %     lambdaP2 = sort(eig(MH(abs((Eo2')*Eo2-2*eye(length(Eo2(1,:)))))));
% %     sigmaP2 = (lambdaP2(1)+lambdaP2(end-1))/2;
% %     if sigmaP2 < 0
% %         sigmaP2
% %         done = 1;
% %         error('!!')
% %     else
%         for i = 2:m
%             for j = 1:i-1
%                 if rand > 0.25*abs(i-j)/(m-1)
%                     A2(i,j) = 1;
%                     A2(j,i) = 1;
%                 end
%             end
%         end
%         t = t+1;
% %     end
% end
load('A2_trial01.mat')
load('x02_trial01.mat')
%trend2 = mean(x0);
%x02 = x0-trend2;
Go1 = graph(A1);
Go2 = graph(A2);
Eo1 = full(incidence(Go1));
Eo2 = full(incidence(Go2));
n1 = numedges(Go1);
n2 = numedges(Go2);
x02 = zeros(n2,1);
col1 = 1;
col2 = 1;
while col1 <= n1 && col2 <= n2
    if norm(Eo1(:,col1) - Eo2(:,col2)) == 0
        x02(col2) = x01(col1);
        col1 = col1+1;
    end
    col2 = col2+1;
end


% termination conditions and hyperparameters
p.gamma = 6*1e-1; % original: 1e-1
p.failure = 1;
p.kmax = 100;
zeta = 1e-3;


for i = epx_ids

    % constructing graphs Go and G
    eval(strcat('x0 = x0',num2str(i),';'));
    eval(strcat('Ao = A',num2str(i),';'));
    Go = graph(Ao);
    p.n = numedges(Go);
    if p.n ~= length(x0)
        fprintf('Dimension mismatch: x0 and n. of edges are different.\n')
        x0 = abs(10*randn(p.n,1));
    end
    Eo = full(incidence(Go));
    A = abs((Eo')*Eo-2*eye(p.n));
    p.G = graph(A);

    % getting the Metropolis Hastings matrix and etaL
    p.P = MH(A);
    lambdaP = sort(eig(p.P));
    sigmaP = (lambdaP(1)+lambdaP(end-1))/2
    p.etaL = zeta;
    if sigmaP < 0
        p.etaL = sigmaP/(sigmaP-1);
    end

    % topological parameters
    d = sum(A);
    dm = min(d);
    dM = max(d);
    d_m_M = [dm dM]
    xi_under = 1/(1+dM);
    xi_over = 1-dm/(1+dM);
    xi = [xi_under xi_over]
    p.omega = 2*dM/(1+dM);
    omega = p.omega
    dists = distances(p.G);
    p.diam = max(max(dists(:)));
    rho = min(max(dists));
    diam_radius = [p.diam rho]

    % constraints and corresponding paramters
    cD = double.empty(1,0); % in general: (p.n,0)
    cU = double.empty(1,0); % in general: (p.n,0)
    p.alphaD = 0.9;
    p.alphaU = 0.95;
    p.amplD = 5; 
    p.amplU = 7;

    % computing etaH
    % Note: computing etaH is a matter of knowing the minimum values of
    % both cD and cU functions over k
    p.etaH = 1-min(cU_(p,1),cD_(p,1))/(p.omega*norm(x0,'Inf'));
    eta_L_H = [p.etaL p.etaH]
    
    % running the water distribution protocol
    eta = double.empty(1,0);
    W = double.empty(1,0);
    dxstar = double.empty(1,0);
    [xx,eta,cD,cU,W,dxstar,kbar] = aacdyn(p,x0,eta,cD,cU,W,dxstar);
    p.kbar = kbar;
    x_final = xx(:,end)
    %kbar = length(eta)
    kk = (0:p.kmax);

    % convergence indexes
    r_over = 1-((1-p.etaH)*xi_under)^rho;
    r_under = 1-(xi_over+(1-xi_over)*p.etaH)^rho;
    r_hat = 1-(1+dm)^-rho;
    r_under_hat_over = [r_under r_hat r_over]
    R = p.diam*(1+(dM-dm)/2)^rho
    
    %% figures
    if i == 1
        plotgraph(ss,Go,p.G,1)
    else
        plotgraph(ss,Go,p.G,0)
    end
    plottraj(ss,p,kk,xx)
    plotW(ss,p,kk(1:end-1),W,i)
    ploteta(ss,p,kk(1:end-1),eta)
    plotnovio(ss,p,kk(1:end-1),cD,cU,dxstar)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% topology plot (graph: H)
function [] = plotgraph(ss,Ho,H,numLables)
    if ss.graphshow
        figure('Position',ss.fpos)
        Ho.Edges.ids = (1:numedges(Ho))';
        Ecolo = rand(numedges(Ho),3);
        hHo = plot(Ho,'LineWidth',1.25*ss.lw,'MarkerSize',10,...
            'NodeColor',[0 0 0],'EdgeColor',Ecolo,...
            'EdgeAlpha',0.9,'NodeLabel',{});
        if numLables 
            hHo.EdgeLabel = Ho.Edges.ids;
            hHo.EdgeFontSize = ss.ftsz*0.8;
            hHo.EdgeFontName = 'Times';
            hHo.Interpreter = 'Latex';
        else
            hHo.LineWidth = hHo.LineWidth/1.5;
        end
        axis equal
        
        figure('Position',ss.fpos)
        Ecol = zeros(numedges(H),3);
        hH = plot(H,'LineWidth',1.25*ss.lw,'MarkerSize',10,...
            'NodeColor',Ecolo,'EdgeColor',Ecol,...
            'EdgeAlpha',1,'NodeLabel',{});
        if numLables 
            hH.NodeLabel = Ho.Edges.ids;
            hH.NodeFontSize = ss.ftsz*0.8;
            hH.NodeFontName = 'Times';
            hH.Interpreter = 'Latex';
        else
            hH.LineWidth = hH.LineWidth/1.75;
        end
        axis equal
    end
end

% trajectories plot
function [] = plottraj(ss,p,kk,xx)
    figure('Position',ss.fpos)
    grid on
    hold on
    x0 = xx(:,1);
    n = length(x0);
    for i = 1:n
        trajplot = stairs(kk,xx(i,:),'LineWidth',ss.lw);
    end
    meanx0plot = plot([kk(1) kk(end)],mean(x0)*[1 1],'--k','LineWidth',0.75*ss.lw);
    minx0 = floor(min(x0));
    maxx0 = ceil(max(x0));
    while mod(minx0,5) ~= 0
        minx0 = minx0-1;
    end
    while mod(maxx0,5) ~= 0
        maxx0 = maxx0+1;
    end
    yax = (minx0:1:maxx0);
    kbarplot = plot((p.kbar-1)*[1 1], [minx0 maxx0],'k','LineWidth',0.75*ss.lw);
    set(gca,'TickLabelInterpreter',ss.in)
    xticks([0 25 50 75 100])
    xtickangle(0)
    yticks(yax) %([minx0 0 maxx0])
    ytickangle(0)
    ylim([minx0 maxx0])
    xlabel('$k$','interpreter',ss.in,'FontSize',ss.ftsz)
    ylabel('$x_i(k)$','Interpreter',ss.in,'FontSize',ss.ftsz)
    ax = gca;
    ax.XAxis.FontSize = ss.ftsz;
    ax.YAxis.FontSize = ss.ftsz;
    plots = [kbarplot]; %meanx0plot 
    strmean = strcat('$ n^{-1} \sum_{j=1}^{n} x_{j}(0) = ',...
        num2str(mean(x0)),'$');
    strkbar = strcat('$ \bar{k} =', num2str(p.kbar-1), '$');
    legend(plots,{strkbar}, ... % strmean, 
        'interpreter',ss.in, ...
        'FontSize',ss.ftszleg);
end

% Lyapunov function plot
function [] = plotW(ss,p,kk,W,i)
    global step
    figure('Position',ss.fpos)
    grid on
    hold on
    gammaplot = plot([0 max(kk)],log(p.gamma)/log(10)*[1 1],'--k','LineWidth',ss.lw);
    Lyapplot = stairs(kk,log(W)/log(10),'color',ss.mygreen,'LineWidth',ss.lw);
    maxy = ceil(max(log(W)/log(10)));
    miny = floor(min(log(W)/log(10)));
    while mod(maxy,step) ~= 0
        maxy = maxy+1;
    end
    while mod(miny,step) ~= 0
        miny = miny-1;
    end
    kbarplot = plot((p.kbar-1)*[1 1],[miny maxy],...
        'k','LineWidth',0.75*ss.lw);
    set(gca,'TickLabelInterpreter',ss.in);
    xticks([0 25 50 75 100])
    xtickangle(0)
    if i == 1
        yticks(miny:1:maxy)
    end
    if i == 2
        yticks(-35:5:5)
    end
    ytickangle(0)
    ylim([miny maxy])
    xlabel('$k$','interpreter',ss.in,'FontSize',ss.ftsz)
    ax = gca;
    ax.XAxis.FontSize = ss.ftsz;
    ax.YAxis.FontSize = ss.ftsz;
    plots = [Lyapplot gammaplot kbarplot];
    strkbar = strcat('$ \bar{k} =', num2str(p.kbar-1), '$');
    legend(plots,{'$\mathrm{Log}(W(k))$', '$\mathrm{Log}(\gamma)$', strkbar}, ...
        'interpreter',ss.in, ...
        'FontSize',ss.ftszleg);
end

% eta plot
function [] = ploteta(ss,p,kk,eta)
    figure('Position',ss.fpos)
    grid on
    hold on
    etaLplot = plot([0 max(kk)],p.etaL*[1 1],'b','LineWidth',1.5*ss.lw);
    etaHplot = plot([0 max(kk)],p.etaH*[1 1],'r','LineWidth',1.5*ss.lw);
    etaplot = stairs(kk,eta,'color',ss.mygreen,'LineWidth',ss.lw);
    max_eta = max([p.etaH eta]);
    kbarplot = plot((p.kbar-1)*[1 1],[p.etaL max_eta],...
        'k','LineWidth',0.75*ss.lw);
    set(gca,'TickLabelInterpreter',ss.in);
    xticks([0 25 50 75 100])
    xtickangle(0)
    yticks([0 0.2 0.4 0.6 0.8])
    ytickangle(0)
    ylim([0 max_eta])
    xlabel('$k$','interpreter',ss.in,'FontSize',ss.ftsz)
    ax = gca;
    ax.XAxis.FontSize = ss.ftsz;
    ax.YAxis.FontSize = ss.ftsz;
    plots = [etaHplot etaplot etaLplot kbarplot];
    strkbar = strcat('$ \bar{k} =', num2str(p.kbar-1), '$');
    legend(plots,{'$\eta_{H}$', '$\eta(k)$', '$\eta_{L}$', strkbar}, ...
        'interpreter',ss.in, ...
        'FontSize',ss.ftszleg, ...
        'location','northeast');
end

% no-constraints-violation plot
function [] = plotnovio(ss,p,kk,cD,cU,dxstar)
    figure('Position',ss.fpos)
    grid on
    hold on
    cDplot = stairs(kk,-cD,'b','LineWidth',1.5*ss.lw);
    cUplot = stairs(kk,cU,'r','LineWidth',1.5*ss.lw);
    dxplot = stairs(kk,dxstar,'color',ss.mygreen,'LineWidth',ss.lw);
    kbarplot = plot((p.kbar-1)*[1 1],[p.amplU -p.amplD],...
        'k','LineWidth',0.75*ss.lw);
    set(gca,'TickLabelInterpreter',ss.in);
    xticks([0 25 50 75 100])
    xtickangle(0)
    %yticks([0 0.2 0.4 0.6 0.8])
    %ytickangle(0)
    ylim([-p.amplD p.amplU])
    xlabel('$k$','interpreter',ss.in,'FontSize',ss.ftsz)
    ax = gca;
    ax.XAxis.FontSize = ss.ftsz;
    ax.YAxis.FontSize = ss.ftsz;
    plots = [cUplot dxplot cDplot kbarplot];
    strkbar = strcat('$ \bar{k} =', num2str(p.kbar-1), '$');
    legend(plots,{'$c_{U}(k)$', '$\delta x^{\star}(k)$', '$-c_{D}(k)$', strkbar}, ...
        'interpreter',ss.in, ...
        'FontSize',ss.ftszleg);
end

% Metropolis-Hastings matrix
function P = MH(A)
    d = sum(A);
    n = length(d);
    P = zeros(n,n);
    for i = 2:n
        for j = 1:i-1
            if A(i,j) == 1
                P(i,j) = 1/(1+max(d(i),d(j)));
                P(j,i) = P(i,j);
            end
        end
    end
    for i = 1:n
        P(i,i) = 1-sum(P(i,:));
    end
end

% adaptive average consensus dynamics
function [xx,eta,cD,cU,W,dxstar,kbar] =...
    aacdyn(p,xx,eta,cD,cU,W,dxstar)
    kbar = p.kmax;
    kbar_found = 0;
    for k = 1:p.kmax
        % k
        xk = xx(:,k);
        xMk = max(xk); %xMk = MCP(p,xk);
        xmk = min(xk); %xmk = -MCP(p,-xk);
        xk_ninf = max(abs(xmk),abs(xMk));
        if xMk ~= max(xk) || xmk ~= min(xk)
            error('Max consensus does not work correctly (post).')
        end
        W = [W xMk-xmk];
        if W(end) <= p.gamma && ~kbar_found
            kbar_found = 1;
            fprintf(['K bar was found at iteration ',num2str(k),'\n'])
            kbar = k;
        end
        cD = [cD cD_(p,k)];
        cU = [cU cU_(p,k)];
        ck = min(min(cD(:,end)),min(cU(:,end)));
        eta = [eta max(p.etaL,1-ck/(p.omega*xk_ninf))];
        xk1 = eta(end)*xk+(1-eta(end))*p.P*xk;
        xx(:,k+1) = xk1;
        dx = xk1-xk;
        [dx_max,jmax] = max(dx);
        [dx_min,jmin] = min(dx);
        jj = jmin;
        if dx_max > -dx_min
            jj = jmax;
        end
        dxstar = [dxstar sign(dx(jj))*abs(dx(jj))];
    end
end

% cD function
function value = cD_(p,k)
    %value = p.amplD*(1-p.alphaD^k)*(1-(p.alphaD^k)*abs(cos(k/10)));
    value = p.amplD;
    if p.failure && 5 <= k && k < 10
        value = p.amplD/10;
    end
    if p.failure && 15 <= k && k < 65 
        value = p.amplD/50;
    end
end

% cU function
function value = cU_(p,k)
    value = p.amplU*(1-p.alphaU^k)*...
        (1+(p.alphaU^k)*abs(cos((k-1)/10)));
    if p.failure && 11 <= k && k < 25
        value = p.amplU/10;
    end
end

% maxconsensus dynamics
function yM = MCP(p,y0)
    n = length(y0);
    yy = zeros(n,1+p.diam);
    yy(:,1) = y0;
    for l = 1:p.diam
        for ups = 1:n
            plot(p.G)
            N_ups = neighbors(p.G,ups);
            sat_ups = yy(ups,l);
            for j_ = 1:length(N_ups)
                j = N_ups(j_);
                if yy(j,l) > sat_ups
                    sat_ups = yy(j,l);
                end
                yy(ups,l+1) = sat_ups;
            end
        end
    end
    yM = yy(1,end);
    for ups = 2:n
        if yy(ups,end) ~= yM
            error('Max consensus does not work correctly (pre).')
        end
    end
end


