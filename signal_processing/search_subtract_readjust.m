function [est_ampls, est_tau_vect,cir] = search_subtract_readjust(rec_sig,w,MPC,Fs,plot_opt)
% Iteratively searches the multipath components from the received
% signal, using the refference waveform w, in order to estimate the propagation
% channel's impulse response.
% The algorithm finds the largest peak and its sample index of the absolute
% values of the matched filter's output, then the estimated delayed version
% of the refference signal is subtracted from the received signal, and the
% algortihm is repeated until all the multipath components are estimated.
%
% [est_ampls, est_tau_vect,cir] = search_subtract_readjust(rec_sig,w,MPC,Fs)
%
% Input arguments
%
% rec_sig       : The received signal affected by noise column vector
% w             : The refference waveform row vector
% MPC           : The number of multipath components
% Fs            : The sampling frequency
% plot_opt      : The plotting option. if 1 then plot, if 0 the don't plot
%
% Output arguments
%
% est_ampls     : The total estimated amplitude vector
% est_tau_vect  : The total estimated time delay vector
% cir           : The channel impulse response
%%
% Example 1
% Trep = 50e-9;
% % The sampling frequency
% Fs = 40e9;
% % The total samples of the signal
% Ntot = fix(Trep*Fs);
% % The memory for the received signal
% r = zeros(1,Ntot);
% % The sample vector of Channel Impulse Response
% sample_vect = [850 1350 1500 1550 1650];
% % Delays vector of CIR in seconds
% tau_vect= sample_vect/Fs
% % The amplitude vector in mV
% ampl_vect = [1.3 0.95 0.51  0.33 0.28]*1e-3
% % Memory for CIR
% h = zeros(1,Ntot);
% h(sample_vect) = ampl_vect;
% % The original UWB waveform
% % The second derivate of Gaussian pulse
% % The pulse width corresponding to a central frequency of 6.6667 GHz
% pw = 0.15e-9;
% % Signal time duration
% t = [-6*pw:1/Fs:6*pw];
% % The refference UWB waveform
% w = zeros(1,length(t));
% w = (1/sqrt(2*pi))*(1-(t/pw).^2).*exp(0.5*(-(t/pw).^2));
% % The refference signal filtered with the CIR
% ref_sig = filter(w,1,h);
% rs_len = length(ref_sig);
% % The signal to noise ratio
% rec_sig = awgn(ref_sig,5,'measured')';
% MPC = length(tau_vect);
%[est_ampls, est_tau_vect] = search_subtract_readjust(rec_sig,w,MPC,Fs,1)
%--------------------------------------------------------------

%% Initialization of the used variables
% Matrix of delayed versions of refference waveform
w_matr = [];
% The total vector of the estimated amplitudes
est_ampls = zeros(1,MPC);
% The total vector of the estimated time delays
est_tau_vect = zeros(1,MPC);
%%
% Total length of the original waveform and received signal
w_len = length(w);
rs_len = length(rec_sig);
% The readjusted signal being the restult of the
% original received signal and the estimated delayed
% version of w signal
adj_sig = rec_sig;
%% Supposing the total number of the multipath components MPC is known
for k=1:MPC
    %% Constructing the matched filter, the refference signal being the w waveform
    match_filt_output = filter(flipud(w),1,adj_sig);
    % Finding the largest peak in the absolute value of the matched
    % filter output
    % Finding all the peaks in the filter's output
    [p,l] = findpeaks(abs(match_filt_output));
    % p_amp - the maxim peak's amplitude
    % loc - its location in the peaks locations vector l
    [p_amp,loc] = max(p);
    % The delays sample index, compensating with the matched
    % filter's delay
    tau_idx = l(loc) - w_len +1;
    % The total vector of the estimated time delays
    est_tau_vect(k) = tau_idx/Fs;
    % Constructing the delayed version of ideal waveform
    w_tau_idx = [zeros(1,tau_idx) w zeros(1,rs_len - w_len -tau_idx)]';
    % The matrix of the estimated delayed verions of w
    w_matr = [w_matr w_tau_idx];
    % Estimating the amplitude of the path using the inverse
    % autocorrelation matrix of the delayed verion of w, multiplied
    % by the correlation between the received signal and delayed
    % version of w
    ampl_tau_idx = (1/(w_tau_idx'*w_tau_idx))*(w_tau_idx'*rec_sig);
    est_ampls(k) = ampl_tau_idx;
    % The readjust signal is constructed by he subtraction of each estimated
    % multipath component from the received signal
    adj_sig = rec_sig;
    for path=1:k
        adj_sig = adj_sig - est_ampls(path).*w_matr(:,path);
    end
end
%% The channel impulse response
cir = zeros(rs_len,1);
for i=1:length(est_ampls)
    cir = cir+ est_ampls(i).*w_matr(:,i);
end


if plot_opt == 1
    figure;
    plot(cir(3000:end));grid on; axis tight; 
    xlabel(' # Samples');ylabel('Amplitude [V]'); 
    title('The channel impulse response');
    %% Plotting the multipath components of the received signal
    sig_ww = est_ampls.*w_matr;
    figure;
    plot(sig_ww);
    grid on; axis tight;
    xlabel(' # Samples');ylabel('Amplitude [V]');
    title('The multipath components');
else
end
end