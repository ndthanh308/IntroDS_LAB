
tf = 7200;
dt = 0.1;
Nt = floor(tf/dt);

time_sec = [1:1:Nt].*dt;

params.PosIdx  = 1:3;
params.VelIdx  = 4:6;
params.OrnIdx  = 7:9;
params.Bac1Idx = 10:12;
params.Bgy1Idx = 13:15;
params.Bac2Idx = 16:18;
params.Bgy2Idx = 19:21;

params.num_states = 21;

[DCM_IMU2FRD_A, DCM_IMU2FRD_B] = IMU2FRDDCM('Titan');


heading_p0 = [0.5, 1, 2, 4, 6]./3;
nH = length(heading_p0);

% gyro bias
gyro_bias_std_dphr = 0.005;

% gyro ARW
gyro_arw_dprthr    = 0.005;

gyro_bias_rps = deg2rad(0.005)/3600;

dph2rps   = (pi/180)/3600; % degrees per hour to radians per sec
dprh2rprs = (pi/180)/sqrt(3600); % degrees per root hour to rad per root sec
ug2mpss   = 1e-6*9.8;

Phdg = zeros(nH,2,Nt);

for idx = 1:2

        for hdx = 1:nH
            % update gyro parameters
            tau = 30*60;
            MGNC_params.Nav.Filter.Config.imu.gyro_bias_repeat = gyro_bias_std_dphr * dph2rps;
            MGNC_params.Nav.Filter.Config.imu.gyro_bias_Qpsd   = (MGNC_params.Nav.Filter.Config.imu.gyro_bias_repeat)^2 * 2 / tau;
            MGNC_params.Nav.Filter.Config.imu.gyro_ang_rw      = gyro_arw_dprthr * dprh2rprs;
            % update accel parameters
            MGNC_params.Nav.Filter.Config.imu.acc_bias_repeat  = 125/3 * ug2mpss;
            MGNC_params.Nav.Filter.Config.imu.acc_bias_Qpsd    = (MGNC_params.Nav.Filter.Config.imu.acc_bias_repeat)^2 * 2 / tau;

            % initialize covariance
            P0 = zeros(params.num_states, params.num_states);
            P0(params.PosIdx,params.PosIdx) = 0.01^2;
            P0(params.VelIdx,params.VelIdx) = 0.01^2;
            P0(params.OrnIdx,params.OrnIdx) = eye(3)*deg2rad(heading_p0(hdx))^2;

            % accel biases
            P0(params.Bac1Idx,params.Bac1Idx) = MGNC_params.Nav.Filter.Config.imu.acc_bias_repeat^2;
            P0(params.Bac2Idx,params.Bac2Idx) = MGNC_params.Nav.Filter.Config.imu.acc_bias_repeat^2;
            % gyro biases
            P0(params.Bgy1Idx,params.Bgy1Idx) = MGNC_params.Nav.Filter.Config.imu.gyro_bias_repeat^2;
            P0(params.Bgy2Idx,params.Bgy2Idx) = MGNC_params.Nav.Filter.Config.imu.gyro_bias_repeat^2;

            Ptmp = zeros(params.num_states, params.num_states, Nt);

            Pk = P0;

            % simulate
            for tdx = 1:Nt
                % propagate
                [Fk, Qk] = ComputeSTMLinCov(params, MGNC_params, dt);
                Pk       = Fk * Pk * Fk' + Qk;

                % ZVU
                [~, Hp]  = MeasurementModelAbsoluteVelocity(zeros(params.num_states), params.PosIdx, 3);
                [~, Hv]  = MeasurementModelAbsoluteVelocity(zeros(params.num_states), params.VelIdx, 3);
                Rp       = eye(3) * MGNC_params.Nav.Filter.Config.zvel_pos_std_m^2;
                Rv       = eye(3) * MGNC_params.Nav.Filter.Config.zvel_vel_std_mps^2;

                R = blkdiag(Rp, Rv);
                H = [Hp; Hv;];       

                % gyrocompass
                if (1)
                    % sensitivity wrt orientation
                    Horn =  - skew(MGNC_params.Nav.Filter.Config.cb_omega_tof_rps);
                    % sensitivity wrt gyro biases
                    Hbgy =  - eye(3) * MGNC_params.Nav.Filter.Config.imu.DCM_imu_a_to_mob;

                    Hgc = zeros(1, params.num_states);
                    Hgc(1, params.OrnIdx) = Horn(2,:);  % east channel
                    Hgc(1, params.Bgy1Idx) = Hbgy(2,:); % east channel

                    Rgc = (MGNC_params.Nav.Filter.Config.imu.gyro_ang_rw)^2 / dt;

                    H = [H; Hgc];
                    R = blkdiag(R, Rgc);           

                    if (idx == 2)
                        % sensitivity wrt orientation
                        Horn =  - skew(MGNC_params.Nav.Filter.Config.cb_omega_tof_rps);
                        % sensitivity wrt gyro biases
                        Hbgy =  - eye(3) * MGNC_params.Nav.Filter.Config.imu.DCM_imu_b_to_mob;

                        Hgc = zeros(1, params.num_states);
                        Hgc(1, params.OrnIdx)  = Horn(2,:);  % east channel
                        Hgc(1, params.Bgy2Idx) = Hbgy(2,:);  % east channel

                        Rgc = (MGNC_params.Nav.Filter.Config.imu.gyro_ang_rw)^2 / dt;

                        H = [H; Hgc];
                        R = blkdiag(R, Rgc);                  

                    end

                end

                % TODO: nullspace

                Pk = MeasurementUpdateLinCov(Pk, H, R);

                Ptmp(:,:,tdx) = Pk;            
            end
            Phdg(hdx,idx,:) = Ptmp(params.OrnIdx(3), params.OrnIdx(3), :);        
        end
end

 PS = PLOT_STANDARDS();
 %% change default font for figs ( old default == Yu Mincho )
 PS.DefaultFont = 'Arial';
 PS.AxisNumbersFontName = 'Arial';
 PS.AxisFont = 'Arial';
 PS.LegendFont = 'Arial';
 PS.TitleFont = 'Arial';
 PS.PlotTextFont = 'Arial';

 fig = figure('visible', 'on', 'units','normalized','outerposition',[0 0 1 1]);
 grid on; grid minor;
 fig_comps.fig = gcf;
 hold on;

hdx = 1;
fig_comps.p1 = plot(time_sec./60, 3*rad2deg(sqrt(squeeze(Phdg(hdx,1,:)))), '-', 'Color', PS.MyBlack,'LineWidth', 4.0, 'DisplayName', 'Single IMU');
fig_comps.p2 = plot(time_sec./60, 3*rad2deg(sqrt(squeeze(Phdg(hdx,2,:)))), '-', 'Color', PS.MyGrey1,'LineWidth', 4.0, 'DisplayName', 'Dual IMU');

xlabel('Initialization Time (minutes)');
ylabel('Heading Uncertainty (deg, 3\sigma)');
legend('show');
legend('AutoUpdate','off')

annotation(gcf,'rectangle', [0.1306875 0.355179704016913 0.774520833333333 0.567653276955604], 'FaceColor',[1 0 0], 'FaceAlpha', 0.25);

hdx = 2;
fig_comps.p3 = plot(time_sec./60, 3*rad2deg(sqrt(squeeze(Phdg(hdx,1,:)))), '--', 'Color', PS.MyBlack,'LineWidth', 4.0, 'DisplayName', sprintf('tau = %i hour', heading_p0(hdx)/3600));
fig_comps.p4 = plot(time_sec./60, 3*rad2deg(sqrt(squeeze(Phdg(hdx,2,:)))), '--', 'Color', PS.MyGrey1,'LineWidth', 4.0, 'DisplayName', sprintf('tau = %i hour', heading_p0(hdx)/3600));

hdx = 3;
fig_comps.p5 = plot(time_sec./60, 3*rad2deg(sqrt(squeeze(Phdg(hdx,1,:)))),':', 'Color', PS.MyBlack,'LineWidth', 4.0, 'DisplayName', sprintf('tau = %i hour', heading_p0(hdx)/3600));
fig_comps.p6 = plot(time_sec./60, 3*rad2deg(sqrt(squeeze(Phdg(hdx,2,:)))),':', 'Color', PS.MyGrey1,'LineWidth', 4.0, 'DisplayName', sprintf('tau = %i hour', heading_p0(hdx)/3600));

hdx = 4;
fig_comps.p7 = plot(time_sec./60, 3*rad2deg(sqrt(squeeze(Phdg(hdx,1,:)))), '-.', 'Color', PS.MyBlack,'LineWidth', 4.0, 'DisplayName', sprintf('tau = %i hour', heading_p0(hdx)/3600));
fig_comps.p8 = plot(time_sec./60, 3*rad2deg(sqrt(squeeze(Phdg(hdx,2,:)))), '-.', 'Color', PS.MyGrey1,'LineWidth', 4.0, 'DisplayName', sprintf('tau = %i hour', heading_p0(hdx)/3600));


plot([time_sec(1), 7200]./60, [1.2, 1.2], '-.r', 'LineWidth', 2.0);
plot([120, 120], [0, 1.2], '-.r', 'LineWidth', 2.0);

legend('Location','southwest');
style = hgexport('readstyle', 'PowerPoint');
hgexport(gcf, 'temp', style, 'applystyle', true)
% STANDARDIZE_FIGURE(fig_comps);


function Pxx = MeasurementUpdateLinCov(Pxx,H,R)
    S = H*Pxx*H' + R;
    K = Pxx*H'/S;
    tmp = eye(length(Pxx)) - K*H;
    Pxx = tmp*Pxx*tmp' + K*R*K';   
end

function [Fk, Qk] = ComputeSTMLinCov(params, MGNC_params, stm_dt_sec)

    % mean acceleration since previous epoch
    acc_ned_mps2 = [0; 0; 1.35];
    % central body rotation rate skew
    cb_skew_omega         = skew(MGNC_params.Nav.Filter.Config.cb_omega_tof_rps);                          
    cb_skew_omega_squared = cb_skew_omega * cb_skew_omega;

    % compute state transition matrix   
    I = eye(3);    
    F12 = I;  
    F21 = -1 * cb_skew_omega_squared;
    F22 = -2 * cb_skew_omega; 
    F33 = -1 * cb_skew_omega;
    
    R_imu_to_tof = eye(3) * MGNC_params.Nav.Filter.Config.imu.DCM_imu_a_to_mob;
    
    % continuous time STM
    Fcon = zeros(params.num_states);
    Fcon(params.PosIdx,params.VelIdx)     = F12; % velocity into position
    Fcon(params.VelIdx,params.PosIdx)     = F21; % centrifugal acceleration
    Fcon(params.VelIdx,params.VelIdx)     = F22; % coriolis effect
    Fcon(params.VelIdx,params.OrnIdx)     = -skew(acc_ned_mps2);
    Fcon(params.OrnIdx,params.OrnIdx)     = F33;
    Fcon(params.VelIdx,params.Bac1Idx)     = -1 * R_imu_to_tof;
    Fcon(params.OrnIdx,params.Bgy1Idx)     = -1 * R_imu_to_tof;
    tau = 30*60;
    Fcon(params.Bac1Idx,params.Bac1Idx) = -1/tau *  I;
    Fcon(params.Bac2Idx,params.Bac2Idx) = -1/tau *  I;
    Fcon(params.Bgy1Idx,params.Bgy1Idx) = -1/tau *  I;
    Fcon(params.Bgy2Idx,params.Bgy2Idx) = -1/tau *  I;
       
    % discretize using second order Taylor Series (ignoring Fcon dot)
    Fk = eye(size(Fcon)) + ...
         Fcon * stm_dt_sec + ...
         0.5 * Fcon * Fcon * stm_dt_sec * stm_dt_sec;

    % TODO: clean up process noise 
    % compute process noise matrix
    Ian = MGNC_params.Nav.Filter.Config.imu.acc_vel_rw     * MGNC_params.Nav.Filter.Config.imu.acc_vel_rw     * eye(2);
    Iab = MGNC_params.Nav.Filter.Config.imu.acc_bias_Qpsd * I;
    Igb = MGNC_params.Nav.Filter.Config.imu.gyro_bias_Qpsd * I;
    Ign = MGNC_params.Nav.Filter.Config.imu.gyro_ang_rw * MGNC_params.Nav.Filter.Config.imu.gyro_ang_rw * I; 
    
    Ian_ned = R_imu_to_tof * (I * MGNC_params.Nav.Filter.Config.imu.acc_vel_rw^2) * R_imu_to_tof';
    Ign_ned = R_imu_to_tof * Ign * R_imu_to_tof';

    Qcon = zeros(params.num_states);
    Qcon(params.VelIdx,params.VelIdx)             = Ian_ned;
    Qcon(params.OrnIdx,params.OrnIdx)             = Ign_ned;
    Qcon(params.Bac1Idx,params.Bac1Idx) = Iab;
    Qcon(params.Bac2Idx,params.Bac2Idx) = Iab;
    Qcon(params.Bgy1Idx,params.Bgy1Idx) = Igb;
    Qcon(params.Bgy2Idx,params.Bgy2Idx) = Igb;
       
    % discretize
    Qk = Qcon * stm_dt_sec; 

end
