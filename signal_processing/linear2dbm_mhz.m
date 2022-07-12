function dsp_dbm_mhz=linear2dbm_mhz(dsp_lineaire)

% Conversion of PSD values from W/Hz into dBm/MHz 
%
% dsp_dbm_mhz=linear2dbm_mhz(dsp_lineaire)
%
% dsp_lineaire - PSD in W/Hz (or V^2/Hz)
% dsp_dbm_mhz - PSD in dBm/MHz
%
% Exemple :
%
% fe=1e3; te=1/fe; T=100e-3; tv=[0:te:T]; N=length(tv);
% T0=30e-3;N0=fix(T0/te)+1;
% av=[2*ones(1,N0) zeros(1,N-N0)];
% figure;subplot(411);plot(tv,av);xlabel('t [s]');ylabel('A [V]');title('Analyzed signal')
% puiss_temp=te*sum(av.^2);
% disp(['Estimated power in the time domain: ', num2str(puiss_temp)])
% comp_cont=av(1)*T0;
% disp(['Signal mean value: ', num2str(comp_cont)])
% Nfft=1001; Nfft_pos=1+(Nfft-1)/2;
% fv=linspace(0,fe/2,Nfft_pos); dfv=fv(2)-fv(1);
% spectr=(1/dfv)*(1/Nfft)*(abs(fft(av,Nfft)));
% spectr_pos=spectr(1:Nfft_pos);
% disp(['DC spectrum component: ', num2str(spectr_pos(1))])
% subplot(412);plot(fv,spectr_pos);xlabel('f [Hz]');ylabel('A [V/Hz]');title('Onesided signal spectrum')
% dens_spectr=spectr_pos.^2;
% subplot(413);plot(fv,dens_spectr);xlabel('f [Hz]');ylabel('PSD [V^2/Hz]');title('Onesided signal power spectrum')
% puiss_freq=2*dfv*sum(dens_spectr);
% disp(['Estimated power in the spectral domain: ', num2str(puiss_freq)])
% dsp_dbm_mhz=linear2dbm_mhz(dens_spectr);
% subplot(414);plot(fv,dsp_dbm_mhz);xlabel('f [Hz]');ylabel('PSD [dBm/MHz]');title('Onesided signal power spectrum')
% dsp_lineaire=dbm_mhz2linear(dsp_dbm_mhz);
% subplot(413);hold on;plot(fv,dsp_lineaire,'r--');xlabel('f [Hz]');ylabel('PSD [V^2/Hz]')

dsp_lineaire(dsp_lineaire==0)=1e-12;
dsp_dbm_mhz=90+10*log10(dsp_lineaire);
