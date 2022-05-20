function [ New_A_network ] = construct_mutation_network( mutation_tumor_fileName,gene_list )
%***********construct co-mutation network for BRCA data set***********

[a,b,c]=xlsread(mutation_tumor_fileName);
gene_name=unique(b(:,1));
sample_name=unique(b(:,2));
[~,e1]=ismember(b(:,1),gene_name);
[~,e2]=ismember(b(:,2),sample_name);
mutation_matrix=zeros(length(gene_name),length(sample_name));
for i=1:size(e1,1)
    i
    mutation_matrix(e1(i,1),e2(i,1))=1;
end


X=mutation_matrix;
Y=mutation_matrix;

tic

%jaccard coefficient of 0-1 matrix
Comutation_network= 1-pdist2(X,Y,'jaccard');

toc


Comutation_network= Comutation_network-diag(diag(Comutation_network));
A_network=Comutation_network;


[z1,z2]=find(triu(A_network)~=0);
z=[z1 z2];


[x,y]=ismember(gene_name,gene_list);

new_z1=y(z1);new_z2=y(z2);
new_z=[new_z1 new_z2];
new_z_ind=new_z1.*new_z2;
new_z(find(new_z_ind==0),:)=[];
z(find(new_z_ind==0),:)=[];

New_A_network=zeros(length(gene_list));
for i=1:size(new_z,1)
    i
   
        
        New_A_network(new_z(i,1),new_z(i,2))=A_network(z(i,1),z(i,2));
         New_A_network(new_z(i,2),new_z(i,1))=A_network(z(i,2),z(i,1));
        
   
end

end

