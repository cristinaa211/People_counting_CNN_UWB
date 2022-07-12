close all; 
clear all

%% Parameters of the baseband UWB pulse 
Tpulse_BB=5e-9;
Fs_BB=4.2e9;
Npulse_BB=fix(Tpulse_BB*Fs_BB);

%% Reading the measured data (reference IR-UWB signal transmitted by cable)
load rfsig_ampli_cable_43dBm_synchro
sigc_amp_ref=double(Channel_1.Data)*Channel_1.YInc; 
sigc_amp_ref=sigc_amp_ref-mean(sigc_amp_ref);
Fs=1/Channel_1.XInc;
Ns_amp=length(sigc_amp_ref);
tv_amp=[0:Ns_amp-1]'/Fs;
figure; plot(tv_amp*1e9,sigc_amp_ref); grid
xlabel('Time [ns]'); ylabel('Amplitude [V]')
title('Amplified reference IR-UWB signal')

%% Envelop detection
sig_ref=sigc_amp_ref; sig_ref=sig_ref/max(abs(sig_ref)); 
sig_ref_ana=hilbert(sig_ref); sig_ref_env=abs(sig_ref_ana);
sig_ref_env=filter(ones(1,10),1,sig_ref_env);
sig_ref_env=sig_ref_env(6:end); sig_ref_env=sig_ref_env/max(abs(sig_ref_env));

figure; plot(sig_ref); grid
hold on; plot(sig_ref_env,'r');
xlabel('# sample'); ylabel('Normalized amplitude');
legend('Measured IR-UWB signal','IR-UWB signal envelop')
title('Envelop detection (generated reference IR-UWB signal)')

[peaks,locs] = findpeaks(sig_ref_env,'MinPeakHeight',0.75,'MinPeakDistance',100);
Nrec_vect=diff(locs)'; Nrec_est=fix(mean(Nrec_vect))
Trec_est=Nrec_est/Fs

figure; 
plot(sig_ref_env); hold on; stem(locs,peaks,'-or');
grid; ylim([-0.25 1.2]); xlabel('# sample'); ylabel('Normalized amplitude');
legend('IR-UWB signal envelop','Detected peaks')
title('Peaks detection on the signal envelop (generated IR-UWB signal)')

%% PSD estimation
Nfft=4*2^fix(log2(Nrec_est));
[Psig,fv_sig]=pwelch(sigc_amp_ref,hamming(Nrec_est),0,16*Nrec_est,Fs);
Psig_dbm_mhz=linear2dbm_mhz(Psig);

figure; plot(fv_sig*1e-9,Psig_dbm_mhz); grid
xlabel('Frequency [GHz]'); ylabel('Amplitude [dBm/MHz]')
title('Power spectral density of the generated IR-UWB signal')

%% Baseband shifting of the reference signal 
fc=7.25e9; 
Fpass=[6.25e9 8.25e9];
sigc_ref=bandpass(sigc_amp_ref,Fpass,Fs);

Fpass=1.5e9;
sigc_ref_mix=sigc_ref.*cos(2*pi*fc*tv_amp);
sigc_ref_BB=lowpass(sigc_ref_mix,Fpass,Fs);
sigc_ref_BB_I=sigc_ref_BB(1:Nrec_est);
sigc_ref_mix=sigc_ref.*sin(2*pi*fc*tv_amp);
sigc_ref_BB=lowpass(sigc_ref_mix,Fpass,Fs);
sigc_ref_BB_Q=sigc_ref_BB(1:Nrec_est);
sigc_ref_BB=abs(sigc_ref_BB_I+j*sigc_ref_BB_Q);
sigc_ref_BB=sigc_ref_BB/max(sigc_ref_BB);
figure; plot(sigc_ref_BB); grid; 
xlabel('# sample'); ylabel('Normalized amplitude');
title('Reference baseband signal on a pulse period')

%% Extraction of the reference signal on the time interval Tpulse = 5 ns
[valmax,idxmax]=max(sigc_ref_BB);
Npulse=fix(Tpulse_BB*Fs)
idx1=max([1 idxmax-Npulse]); idx2=min([Nrec_est idxmax+Npulse]);
figure; plot(sigc_ref_BB); grid; hold on; 
plot([idx1:idx2],sigc_ref_BB(idx1:idx2),'r');
xlabel('# sample'); ylabel('Normalized amplitude');
legend('Measured reference signal','Reference signal limited on 2*Tpulse')
sigc_ref_BB=sigc_ref_BB(idx1:idx2); Np=length(sigc_ref_BB); kvect=[1:Np];
spdf=sigc_ref_BB.^2/norm(sigc_ref_BB)^2;
km=fix(sum(kvect.*spdf'))
figure; plot(sigc_ref_BB); grid; hold on; 
plot([km-fix(Npulse/2):km+fix(Npulse/2)],sigc_ref_BB(km-fix(Npulse/2):km+fix(Npulse/2)),'r');
xlabel('# sample'); ylabel('Normalized amplitude');
legend('Reference signal limited on 2*Tpulse','Reference signal limited on Tpulse')
sigc_ref_BB=sigc_ref_BB(km-fix(Npulse/2):km+fix(Npulse/2));
sigc_ref_BB=sigc_ref_BB-min(sigc_ref_BB);
sigc_ref_BB=sigc_ref_BB/max(sigc_ref_BB);
figure; plot(sigc_ref_BB); grid; 
xlabel('# sample'); ylabel('Normalized amplitude');
title('Reference signal on the time interval Tpulse')

%% Reading the measured data (wireless transmitted IR-UWB signal)
load rfsig_ampli_ant_0dBm_synchro3
sigc_amp_wrls=double(Channel_1.Data)*Channel_1.YInc; 
sigc_amp_wrls=sigc_amp_wrls-mean(sigc_amp_wrls);
Fs=1/Channel_1.XInc;
Ns_amp=length(sigc_amp_wrls); tv_amp=[0:Ns_amp-1]'/Fs;
figure; plot(tv_amp*1e9,sigc_amp_wrls); grid
xlabel('Time [ns]'); ylabel('Amplitude [V]')
title('Amplified wireless IR-UWB signal')

%% Baseband shifting of the wireless signal 
fc=7.25e9; 
Fpass=[6.25e9 8.25e9];
sigc_wrls=bandpass(sigc_amp_wrls,Fpass,Fs);
figure; plot(tv_amp*1e9,sigc_wrls); grid
xlabel('Time [ns]'); ylabel('Amplitude [V]')
title('Amplified wireless IR-UWB signal after bandpass filtering')

Fpass=1.5e9;
sigc_wrls_mix=sigc_wrls.*cos(2*pi*fc*tv_amp);
sigc_wrls_BB_I=lowpass(sigc_wrls_mix,Fpass,Fs);
% sigc_wrls_BB_I=sigc_wrls_BB(1:Nrec_est);
sigc_wrls_mix=sigc_wrls.*sin(2*pi*fc*tv_amp);
sigc_wrls_BB_Q=lowpass(sigc_wrls_mix,Fpass,Fs);
% sigc_wrls_BB_Q=sigc_wrls_BB(1:Nrec_est);
sigc_wrls_BB=abs(sigc_wrls_BB_I+j*sigc_wrls_BB_Q);
sigc_wrls_BB=sigc_wrls_BB-min(sigc_wrls_BB);
sigc_wrls_BB=sigc_wrls_BB/max(sigc_wrls_BB);
figure; plot(sigc_wrls_BB); grid; 
xlabel('# sample'); ylabel('Normalized amplitude');
title('Baseband shifted wireless signal')

%% Matched filtering of the BB wireless signal
sigc_wrls_BB_MF=filter(flipud(sigc_ref_BB),1,sigc_wrls_BB);
sigc_wrls_BB_MF=sigc_wrls_BB_MF(fix(Npulse/2):end);
figure; plot(sigc_wrls_BB_MF(1:4000)/max(abs(sigc_wrls_BB_MF))); grid; 
xlabel('# sample'); ylabel('Normalized amplitude');
title('Wireless signal after matched filtering')

%% Applying the search subtract and readjust algorithm 
[est_ampls, est_tau_vect] = search_subtract_readjust(sigc_wrls_BB_MF(1:Nrec_est)/max(sigc_wrls_BB_MF(1:Nrec_est)),sigc_ref_BB',3,Fs,1);



