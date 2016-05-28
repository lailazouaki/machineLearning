TRAINING ALGO

%loading the speech data
%st=indicates the beginning sample number of each speech sequence
%speech=contans the speech samples
fs=16000; %sampling rate in Hz
phoneme='aa'
filename=strcat
('G:\my_thesis\phoneme_timit\train_Dr1_',phoneme,'.mat');
load(filename);%gives wav and st
label=st;
clear st;

fprintf('EXtracting features-mel cepstrum 26 parameters:\n');
train_data=[];
for(i=1:length(label)-1)     
    C=melcepst(wav(label(i):(label(i+1)-1)),fs,'Med',12,256,64);
    train_data{i}=C';
end

%defining the parameters for the HMM
M=1; % number of gaussioan mixtures per state
Q=3; % number of states
cov_type='diag';
left_right=1; % bakis model

fprintf('\nInitializing the HMM.....');
[prior0, transmat0, mixmat0, mu0, Sigma0] =  init_mhmm(train_data{1}, 
Q, M, cov_type, left_right);
    
%The EM algorithm loop
max_iter=30;
thresh=1e-5;
[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
    learn_mhmm(train_data, prior0, transmat0, mu0, Sigma0, mixmat0, 
max_iter,thresh);

pth=strcat('G:\my_thesis\hmmmodels\b',phoneme);
save(pth,'LL','prior1','transmat1','mu1','Sigma1','mixmat1');
fprintf('End of Training...clearing all variables')
beep;
clear all;


RECOGNITION ALGO

phoneme_model='baa';
phoneme_test='ae';

filename=strcat('G:\my_thesis\hmmmodels\',phoneme_model,'.mat');
load(filename);%gives A_,mu_,Sigma_
f=strcat('G:\my_thesis\phoneme_timit\test_Dr1_',phoneme_test,'.mat');
load(f);
label=st;
clear st;

fprintf('\nextracting cepstrum for multiple observations...\n')
test_data=[];
st=[];
for(i=1:length(label)-1)     
   C=melcepst(wav(label(i):(label(i+1)-1)),fs,'Med',12,128,32);
   test_data{i}=C';
end

fprintf('\nStarting the recognition\n')
%recognition 
for(i=1:length(test_data))
 %obs_lik= mk_mhmm_obs_lik(test_data{1}, mu1, Sigma1, mixmat1);
%[path, loglik(i)] = viterbi_path(prior, transmat, B);
loglik(i)=log_lik_mhmm(test_data{i}, prior1, 
transmat1,mixmat1,mu1,Sigma1);
end

rec=max(loglik)

fprintf('End of Testing');

