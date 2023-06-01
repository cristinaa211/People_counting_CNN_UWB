function [kmat,pes]=PCA_data_projection(pen,thrs,plotopt)

% Principal Component Analysis (PCA)
%
% [kmat,pes]=PCA_data_projection(pen,thrs,plotopt);
%
% pen     - matrix of vectors to be projected
% thrs    - if higher than 1, it stands for the dimension of the projection space
%           if lower than 1, it stands for the energy percentage of the largest selected eignevalues of the data covariance matrix 
% plotopt - if equal to 'plot' enables graphical representations
% kmat    - projection matrix
% pes     - matrix of the projected vectors

sigx=cov(pen');
[vectp,valp]=eig(sigx);
vlpr=abs(diag(valp)); [ld,idx]=sort(vlpr,'descend');
vectps=vectp(:,idx);

if thrs<1
    ldn=(norm(ld))^2; ldns=cumsum(ld.^2);
    ldns=ldns/ldn; idxc=find(ldns>=thrs); ns=idxc(1);
else
    ns=thrs;
end

kmat=vectps(:,1:ns).';
pes=kmat*pen;

if strcmp(plotopt,'plot')
    figure; stem(ld,'-ob'); grid; hold on
    stem(ld(1:ns),'-xr')
    xlabel('#'); ylabel('Eigenvalue magnitude')
    legend('Data covariance matrix eigenvalues','Selected eignevalues')
    title('Eigenvalues distribution (PCA)')
    disp(['Eigenvalues vector ld = ', num2str(ld')])
end
