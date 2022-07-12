function [final_features_mean,car_norm] = feature_extraction_curv_dstbin(X,fs,l_cutoff,u_cutoff,norm)

% X        : input data
% fs        : the sampling frequency , default 39e9
% u_cutoff  : the upper cutoff frequency of bandpass filter , default 7.95e9
% l_cutoff  : the lower cutoff frequency of bandpass filter , default 5.65e9
% norm      : if 1 then normalize, if 0 then no

if u_cutoff == 0 && l_cutoff == 0
    u_cutoff = 7.95e9;
    l_cutoff = 5.65e9;
end
if fs == 0
    fs = 39e9;
end

%% Data preprocessing
figure;
imagesc(X);colormap(gray); xlabel('Distance (m)');ylabel('Time (sec)'); title('Original data');
% DC Removal
X = X-mean(X);
% Bandpass filtering
X= bandpass(X,[l_cutoff,u_cutoff],fs);
%fft_sig(X,fs,1);
figure;
imagesc(X);colormap(gray); xlabel('Distance (m)');ylabel('Time (sec)'); title('Band passed data');
% The running average based method
rec_sig = movmean(X,5);
figure;
imagesc(rec_sig);colormap(gray);xlabel('Distance (m)');ylabel('Time (sec)'); title('Reconstructed data');
[m,n] = size(X);

%% Establishing the numbers of scales for curvelet transform
min_scale = 1; max_scale=3;
final_features = [];
final_features_mean  =[];
for j=1
    for i=0:m/50-1
        %% Taking each radar sample of 200 signals
        data_50 =  X(((1+50*i):50*(i+1)),:);
        %% Apply curvelet transform
        C = fdct_wrapping(data_50,1,1,3);
        %% Extracting the features in the coarse layer
        for k=1:32
            % Found out the mean in the coarse layer
            C_coarse = cell2mat(C{1,3}(k));
            C_coarse = C_coarse(:);
            mean_feature(k) = mean(C_coarse);
            % Finding the energy in the coarse layer
            energy_first_coarse(k) = norm(C_coarse).^2;
        end
        energy_feature_coarse = sum(energy_first_coarse);
        mean_feature_coarse = mean(mean_feature);
        %% Find the maximum 5 amplitudes in the fine layer
        celltomat_fine = cell2mat(C{1,1}(1));
        feature_peaks_fineL=findpeaks(celltomat_fine(:),'NPeaks',5);
        feature_peaks_fineL=feature_peaks_fineL';
        %% Finding coefficients in the detail layer
        % and in the reconstructed detail layer by suppressing the low
        % frequency coefficients
        for th=1:16
            % Converting from cell to array of the detail layer
            celltomat_detail = cell2mat(C{1,2}(th));
            celltomat_detail = celltomat_detail(:);
            %% Hard thresholding on energy
            thresh = 0.9;
            energy_coeff = norm(celltomat_detail)^2;
            [csort,idxsort] = sort(abs(celltomat_detail).^2,'descend');
            c_cum = cumsum(csort);
            idx_vect = find(c_cum >= thresh*energy_coeff);
            idx = idx_vect(1);
            c_out = celltomat_detail;
            c_out(idxsort(idx:end)) =0 ;
            %% Computing the energy of the detail layer
            energy_first_detail(th) = energy_coeff;
            energy_reconst_detail(th) = norm(c_out)^2;
        end
        %% Energy of 45 degrees coefficients
        % meaning people getting away from the radar
        energy_45_feature = energy_first_detail(1)+energy_first_detail(16)+energy_first_detail(8)+energy_first_detail(9);
        energy_45_r_feature = energy_reconst_detail(1)+energy_reconst_detail(16)+energy_reconst_detail(8)+energy_reconst_detail(9);
        % Energy of 135 degrees coefficients
        % meaning people aproaching the radar
        energy_135_r_feature = energy_reconst_detail(4)+energy_reconst_detail(5)+energy_reconst_detail(12)+energy_reconst_detail(13);
        energy_135_feature = energy_first_detail(4)+energy_first_detail(5)+energy_first_detail(12)+energy_first_detail(13);
        % Energy of 90 degrees coefficients
        % meaning people standing still in the LOS of the radar
        energy_90_feature = energy_first_detail(6)+energy_first_detail(7)+energy_first_detail(14)+energy_first_detail(15);
        energy_90_r_feature = energy_reconst_detail(6)+energy_reconst_detail(7)+energy_reconst_detail(14)+energy_reconst_detail(15);
        %% Plotting the coefficients in the fine, detail and coarse layer
        %             for k=min_scale:max_scale
        %                 x = C{1,k}{1,1};
        %                 figure;
        %                 subplot(3,1,1);
        %                 imshow(C{1,1}{1,1},[]); title('Finest layer'); xlabel('distance');ylabel('time');
        %                    subplot(3,1,2);
        %                 imshow(C{1,2}{1,1},[]); title('Detail layer'); xlabel('distance');ylabel('time');
        %                    subplot(3,1,3);
        %                 imshow(C{1,3}{1,1},[]); title('Coarse layer'); xlabel('distance');ylabel('time');
        %
        %                 if k==max_scale
        %                     subtitle('Coarse layer ');
        %                 elseif k==min_scale
        %                     subtitle('Finest layer');
        %                 else
        %                     subtitle('Detail layer');
        %                 end
        %                 hold on;
        %             end
        %% Distance bin
        % Hard thresholding curvelet transform reconstruction
        C_d = C;
        [a,b] = size(data_50);
        E = cell(size(C_d));
        for s=1:length(C_d)
            E{s} = cell(size(C_d{s}));
            for w=1:length(C_d{s})
                A = C_d{s}{w};
                E{s}{w} = sqrt(sum(sum(A.*conj(A))) / prod(size(A)));
            end
        end
        Ct = C_d;
        for s = 2:length(C_d)
            for w = 1:length(C_d{s})
                Ct{s}{w} = C_d{s}{w}.* (abs(C_d{s}{w}) > thresh*E{s}{w});
            end
        end
        % The reconstructed signal
        sig_1280_d = ifdct_wrapping(Ct,1,a,b);
        for k=1:50
            sig_1280_rr = rec_sig(k,:);
            sig_1280_dd = sig_1280_d(k,:);
            %% Extracting features based on distance bin
            for i=1:round(b/32)
                distb_32_d = sig_1280_dd(32*(i-1)+1:32*(i));
                distb_32_r = sig_1280_rr(32*(i-1)+1:32*(i));
                energy_32_feature(k,i) = norm(distb_32_d)^2;
                feature_ampl_32(k,i) = max(distb_32_d);
                energy_32_feature_r(k,i) = norm(distb_32_r)^2;
                feature_ampl_32_r(k,i) = max(distb_32_r);
            end
            for i=1:round(b/64)
                distb_64_d = sig_1280_dd(64*(i-1)+1:64*(i));
                distb_64_r = sig_1280_rr(64*(i-1)+1:64*(i));
                energy_64_feature(k,i) = norm(distb_64_d)^2;
                feature_ampl_64(k,i) = max(distb_64_d);
                energy_64_feature_r(k,i) = norm(distb_64_r)^2;
                feature_ampl_64_r(k,i) = max(distb_64_r);
            end
            for i=1:round(b/128)
                distb_128_d = sig_1280_dd(128*(i-1)+1:128*(i));
                distb_128_r = sig_1280_rr(128*(i-1)+1:128*(i));
                energy_128_feature(k,i) = norm(distb_128_d)^2;
                feature_ampl_128(k,i) = max(distb_128_d);
                energy_128_feature_r(k,i) = norm(distb_128_r)^2;
                feature_ampl_128_r(k,i) = max(distb_128_r);
                
            end
            features = [energy_feature_coarse,mean_feature_coarse,feature_peaks_fineL,energy_45_feature, energy_45_r_feature,energy_135_r_feature,energy_135_feature,energy_90_feature ,energy_90_r_feature,energy_32_feature(k,:),feature_ampl_32(k,:),energy_32_feature_r(k,:),feature_ampl_32_r(k,:),energy_64_feature(k,:),feature_ampl_64(k,:), energy_64_feature_r(k,:),feature_ampl_64_r(k,:),energy_128_feature(k,:),feature_ampl_128(k,:),energy_128_feature_r(k,:),feature_ampl_128_r(k,:)];
            final_features = [final_features ; features];
        end
        final_features_mean = [final_features_mean; mean(final_features)];
        
    end
    
end
if norm == 1
    %% Features Normalization
    N = size( final_features_mean,1);
    car= final_features_mean;
    x1 = car(:,1);
    Rx1 =(x1'*x1)/N;
    K1 = diag(sqrt(1./diag(Rx1)));
    Y1 = x1*K1;
    
    %% Second Category
    x2 = car(:,2);
    Rx2 =(x2'*x2)/N;
    K2 = diag(sqrt(1./diag(Rx2)));
    Y2 = x2*K2;
    %% Third category
    
    x3 = car(:,3:7);
    Rx3 =(x3'*x3)/N;
    K3 = diag(sqrt(1./diag(Rx3)));
    Y3 = x3*K3;
    
    %% 4 category
    x4 = car(:,8);
    Rx4 =(x4'*x4)/N;
    K4 = diag(sqrt(1./diag(Rx4)));
    Y4 = x4*K4;
    
    %% cat 5
    x5 = car(:,9);
    Rx5 =(x5'*x5)/N;
    K5 = diag(sqrt(1./diag(Rx5)));
    Y5 = x5*K5;
    
    %% cat 6
    x6 = car(:,10);
    Rx6 =(x6'*x6)/N;
    K6 = diag(sqrt(1./diag(Rx6)));
    Y6 = x6*K6;
    
    %% cat 7
    x7 = car(:,11);
    Rx7 =(x7'*x7)/N;
    K7 = diag(sqrt(1./diag(Rx7)));
    Y7 = x7*K7;
    
    %% cat 8
    x8 = car(:,12);
    Rx8 =(x8'*x8)/N;
    K8 = diag(sqrt(1./diag(Rx8)));
    Y8 = x8*K8;
    
    %% cat 9
    x9 = car(:,13);
    Rx9 =(x9'*x9)/N;
    K9 = diag(sqrt(1./diag(Rx9)));
    Y9 = x9*K9;
    
    %% cat 10
    x10 = car(:,14:(14+round(n/32)-1));
    Rx10 =(x10'*x10)/N;
    K10 = diag(sqrt(1./diag(Rx10)));
    Y10 = x10*K10;
    
    %% cat 11
    x11 = car(:,(14+round(n/32)):(14+2*round(n/32)-1));
    Rx11 =(x11'*x11)/N;
    K11 = diag(sqrt(1./diag(Rx11)));
    Y11 = x11*K11;
    
    
    %% cat 12
    x12 = car(:,(14+2*round(n/32)):(14+3*round(n/32)-1));
    Rx12 =(x12'*x12)/N;
    K12 = diag(sqrt(1./diag(Rx12)));
    Y12 = x12*K12;
    
    %% cat 13
    x13 = car(:,(14+3*round(n/32)):(14+4*round(n/32)-1));
    Rx13 =(x13'*x13)/N;
    K13 = diag(sqrt(1./diag(Rx13)));
    Y13 = x13*K13;
    
    %% cat 14
    x14 = car(:,(14+4*round(n/32)):(14+4*round(n/32)+round(n/64)-1));
    Rx14 =(x14'*x14)/N;
    K14 = diag(sqrt(1./diag(Rx14)));
    Y14 = x14*K14;
    
    %% cat 15
    x15 = car(:,(14+4*round(n/32)+round(n/64)):(14+4*round(n/32)+2*round(n/64)-1));
    Rx15 =(x15'*x15)/N;
    K15 = diag(sqrt(1./diag(Rx15)));
    Y15 = x15*K15;
    
    %% cat 16
    x16 = car(:,(14+4*round(n/32)+2*round(n/64)):(14+4*round(n/32)+3*round(n/64)-1));
    Rx16 =(x16'*x16)/N;
    K16 = diag(sqrt(1./diag(Rx16)));
    Y16 = x16*K16;
    
    
    %% cat 17
    x17 = car(:,(14+4*round(n/32)+3*round(n/64)):(14+4*round(n/32)+4*round(n/64)-1));
    Rx17 =(x17'*x17)/N;
    K17 = diag(sqrt(1./diag(Rx17)));
    Y17 = x17*K17;
    
    %% cat 18
    x18 = car(:,(14+4*round(n/32)+4*round(n/64)):(14+4*round(n/32)+4*round(n/64)+round(n/128)-1));
    Rx18 =(x18'*x18)/N;
    K18 = diag(sqrt(1./diag(Rx18)));
    Y18 = x18*K18;
    
    
    %% cat 19
    
    x19 = car(:,(14+4*round(n/32)+4*round(n/64)+round(n/128)):(14+4*round(n/32)+4*round(n/64)+2*round(n/128)-1));
    Rx19 =(x19'*x19)/N;
    K19 = diag(sqrt(1./diag(Rx19)));
    Y19 = x19*K19;
    
    
    %% cat 20
    x20 = car(:,(14+4*round(n/32)+4*round(n/64)+2*round(n/128)):(14+4*round(n/32)+4*round(n/64)+3*round(n/128)-1));
    Rx20 =(x20'*x20)/N;
    K20 = diag(sqrt(1./diag(Rx20)));
    Y20 = x20*K20;
    
    %% cat 21
    x21 = car(:,(14+4*round(n/32)+4*round(n/64)+3*round(n/128)):(14+4*round(n/32)+4*round(n/64)+4*round(n/128)-1));
    Rx21 =(x21'*x21)/N;
    K21 = diag(sqrt(1./diag(Rx21)));
    matr = x21*K21;
    
    
    car_norm = [Y1 Y2 Y3 Y4 Y5 Y6 Y7 Y7 Y9 Y10 Y11 Y12 Y13 Y14 Y15 Y16 Y17 Y18 Y19 Y20 matr];
    %K = [K1 K2 K3 K4 K5 K6 K7 K8 K9 K10 K11 K12 K13 K14 K15 K16 K17 K18 K19 K20 K21];
    save('k1_p1.mat','K1');
    save('k2_p1.mat','K2');
    save('k3_p1.mat','K3');
    save('k4_p1.mat','K4');
    save('k5_p1.mat','K5');
    save('k6_p1.mat','K6');
    save('k7_p1.mat','K7');
    save('k8_p1.mat','K8');
    save('k9_p1.mat','K9');
    save('k10_p1.mat','K10');
    save('k11_p1.mat','K11');
    save('k13_p1.mat','K13');
    save('k14_p1.mat','K14');
    save('k15_p1.mat','K15');
    save('k16_p1.mat','K16');
    save('k17_p1.mat','K17');
    save('k18_p1.mat','K18');
    save('k12_p1.mat','K12');
    save('k19_p1.mat','K19');
    save('k20_p1.mat','K20');
    save('k21_p1.mat','K21');
end
end