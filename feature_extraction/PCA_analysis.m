clear all
close all

load date_raw.mat
label_vect=X_final(:,294);

Nscrs=4;                  % number of scenarios
Nhmns_max=25;             % maximum number of humans considered in the different scenarios
Nhmns_vect=[0:Nhmns_max]; % vector of possible number of humans considered in the different scenarios
Ncls=length(Nhmns_vect);  % number of classes (each class corresponds to a number of humans)
thrs=0.95                 % energy percentage of the largest selected eignevalues of the data covariance matrix
plotopt='plot';

matr_vect=[];
for kscr=[1 2 3 4]        % considered scenarios 
    for khum=1:Ncls
        Nhmns=Nhmns_vect(khum);
        labelc=str2num([num2str(kscr) num2str(Nhmns)]);
        idxc=find(label_vect==labelc);
        if ~isempty(idxc)
            matr_vectc=[X_final(idxc,1:293).';khum*ones(1,length(idxc))];
            matr_vect=[matr_vect matr_vectc];
        end
    end
end
[kmat_PCA,matr_proj_PCA]=PCA_data_projection(matr_vect(1:end-1,:),thrs,plotopt);

pcol_matr=['bx';'ro';'gs';'m+';'k.';'cd';'bv';'r*';'gp';'m^';'k<';'ch';'b>'];  % type of markers used to plot the different classes
plotted_dims=[1 2 3];                 % couple of 2 or 3 plotted dimensions in the projection space
plotted_Nhmns=[0 1 5 10 15 20];       % set of number of humans selected to be plotted (less than 13)

if (length(plotted_Nhmns)<13)&(length(plotted_dims)==2)
    figure; legend_txt=[];
    for khum=1:length(plotted_Nhmns)
        labelc=plotted_Nhmns(khum)+1;
        legend_txt=[legend_txt '''N_h_u_m = ',num2str(plotted_Nhmns(khum)),''','];
        matr_projc=matr_proj_PCA(:,matr_vect(end,:)==labelc);
        plot(matr_projc(plotted_dims(1),:),matr_projc(plotted_dims(2),:),pcol_matr(khum,:))
        hold on
    end
    xlabel('y_1'); ylabel('y_2'); title(''); grid
    legend_txt=legend_txt(1:end-1);
    eval(['legend(',legend_txt,',''Location'',''Best'')'])
elseif (length(plotted_Nhmns)<13)&(length(plotted_dims)==3)
    figure; legend_txt=[];
    for khum=1:length(plotted_Nhmns)
        labelc=plotted_Nhmns(khum)+1;
        legend_txt=[legend_txt '''N_h_u_m = ',num2str(plotted_Nhmns(khum)),''','];
        matr_projc=matr_proj_PCA(:,matr_vect(end,:)==labelc);
        plot3(matr_projc(plotted_dims(1),:),matr_projc(plotted_dims(2),:),matr_projc(plotted_dims(3),:),pcol_matr(khum,:))
        hold on
    end
    xlabel('y_1'); ylabel('y_2'); zlabel('y_3'); title(''); grid
    legend_txt=legend_txt(1:end-1);
    eval(['legend(',legend_txt,',''Location'',''Best'')'])
end
