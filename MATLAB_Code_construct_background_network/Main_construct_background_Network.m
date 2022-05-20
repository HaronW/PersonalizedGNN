clc
clear
%************************part1:LOAD sample data and network data************************
%*********Case 1:BRCA data**********
unzip('BRCA_normal.zip') 
unzip('BRCA_tumor.zip') 
expression_tumor_fileName = 'BRCA_tumor.txt';
expression_normal_fileName = 'BRCA_normal.txt';
% load('BRCA_mutation_data.mat')

% %*********Case 2:LUSC(1-49)+LUAD data(50-106)**********
% unzip('LUNG_normal.zip') 
% unzip('LUNG_tumor.zip') 
% expression_tumor_fileName = 'LUNG_tumor.txt';
% expression_normal_fileName = 'LUNG_normal.txt';
% load('LUNG_mutation_data.mat')

% *************************tumor data****************************

tumor=importdata(expression_tumor_fileName);
gene_list=tumor.textdata(2:end,1);
Sample_name_tumor=tumor.textdata(1,2:end);
Tumor=tumor.data;

%*************************normal data****************************

[normal,~,name_normal]=importdata(expression_normal_fileName);
Sample_name_normal=normal.textdata(1,2:end);
Normal=normal.data;


%******************network*********************
mutation_tumor_fileName= 'BRCA_mutation_data';%or LUNG_mutation_data
%mutation_tumor_fileName= 'LUNG_mutation_data';

%Code for obtaining Co-mutation network
[ New_A_network ] = construct_mutation_network( mutation_tumor_fileName,gene_list);


%Gene interaction network
load('GIN_network_information.mat')
Network=[];
D=[];
Result=[];
%[~,PPI,~]=xlsread('network_FIsInGene_041709.xlsx');
[x1,y1]=ismember(edge0(:,1),gene_list);
[x2,y2]=ismember(edge0(:,2),gene_list);
y=y1.*y2;
z=[y1 y2];
z(find(y==0),:)=[];
N1=length(gene_list);
[N2,~]=size(z);
%calculate the adjacency matrix of PPI 
Net_adjacent=zeros(N1,N1);
for i=1:N2
    
        Net_adjacent(z(i,2),z(i,1))=1;
        Net_adjacent(z(i,1),z(i,2))=1;
end

New_A_integrated_network=New_A_network.*Net_adjacent;
[z1,z2]=find(triu(New_A_integrated_network)~=0);
Z=[z1 z2];
Edges=gene_list(Z);

for i=1:size(Z,1)
  Network_comutation_score(i,1)=New_A_integrated_network(Z(i,1),Z(i,2));
end


%Output the background network and corresponding network score file

fidw=fopen('background.txt','w');
for i=1:size(Edges,1)
   S=[Edges{i,1} Edges{i,2}];    
    fprintf(fidw,'%s\n',S);end
end
fclose(fidw);


fidw=fopen('comutation.txt','w');
for i=1:size(Edges,1)
   S=Network_comutation_score(i,1);    
    fprintf(fidw,'%d\n',S);end
end
fclose(fidw);



