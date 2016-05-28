database.features=[];
database.nfeatures=[];
database.instruments=[ones(15,1) ; 2*ones(15,1) ; 3*ones(15,1) ; 4*ones(15,1)];
database.classifications=ones(60,1);

filenames=dir('./signaux/*.wav');
database.filenames={};
for i=1:size(filenames,1)
  database.filenames{i,1}=filenames(i).name;
end

save db.mat database