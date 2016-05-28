function database=instrument_classification(partie_1_finie,test_piccolo_flag)

% - partie_1_finie: mettre a 1 une fois qu'on aura termin? la partie
%   sur l'extraction des attributs, sinon mettre a 0 ;
%  
% - test_piccolo_flag: mettre a 1 pour faire la question 7.2, sinon
% mettre a 0
  

% -- Fonction main

% Charger la base de donn?es
load ('db.mat');

if partie_1_finie == 0,
  % Extraction d'attributs ? partir du signal audio
  % Une fois qu'on aura termin? la partie sur l'extraction des attributs
  % Cette ligne n'est plus ex?cut?e pour gagner du temps!
  database.features = compute_features(database);
  save db.mat database
else,
  if test_piccolo_flag == 0
  % Standardisation des attributs
  nfeats=size(database.features,2); % nombre d'attributs
  database.nfeatures = standardize(database.features(:,[1:nfeats]));
  
  % Apprentissage et classification par mod?le gaussien
  database = cross_validation(database, @gaussian_train, @gaussian_classify);

  % Apprentissage et classification par knn 
  database2 = cross_validation(database, @knn_train, @knn_classify);

  % Evaluation
  display_results(database);
  display_results(database2);
  else
    test_piccolo(database);
  end
end;


% -- Fonctions auxiliaires

% Calcule les attributs ? partir d'un signal audio
function features = compute_features(database,file_i)
  N=1024; % taille de la fenetre d'analyse
  if nargin<2
    nbfiles = size(database.filenames,1);
    ifiles=(1:nbfiles);
    features = zeros(nbfiles,20); % 20 attributs
  else
    features=[database.features ; zeros(1,20)];
    ifiles=file_i;
  end

  for i=ifiles
    soundfile = sprintf('signaux/%s',char(database.filenames(i)));
    fprintf('Computing features for file %s...\n', soundfile);
    [s, sr] = wavread(soundfile);
    % Normalisation du signal 
    s = s-mean(s);
    s = s/max(abs(s));
    % Fenetrage du signal
    Frames = enframe(s,N,N/2);
    % calcule des moments spectraux et du spectre (que l'on garde pour
    % le calcul d'autres descripteurs spectraux)
    [sx,Spec] = features_specMoments(Frames,N,sr);
    % tous les autres attributs
    features(i,:) = [features_zcr(s, sr) sx features_specOsc(Spec) features_specSlope(Spec,N,sr)...
                    features_lpcCoef(Frames) features_freqCutoff(Spec,sr,N)...
                     features_mfcc(Frames,Spec,N,sr)];
  end;
%endfunction

% Calcul des param?tres
function zcr = features_zcr(s, sr)
  %disp(size(s));
  zcr = sum(abs(diff(sign(s))))/2/sr;
%endfunction

function [sx,Spec] = features_specMoments(xn,N,sr)
% specMoments - Compute spectral moments features
%
% SYNTAX
% [sx,Spec] = specMoments(xn,N,sr)
%
% ARGUMENTS
% xn - input signal segment or "framed signal matrix" i.e a matrix
%      where each row is a signal frame
% N  - frame length (size(xn,2))
% sr - sampling freq
%
% OUTPUT
% sx=[sc sw sa sf]
% sc - Spectral centroid
% sw - Spectral width
% sa - Spectral asymmetry
% sf - Spectral flatness
% Spec - Frames spectrum

% -- Compute mag spectrum
xn = xn.*(ones(size(xn,1),1)*hamming(N).');
Spec = fft(xn.').'; clear xn
Spec = Spec(:,1:N/2+1);
Ak = abs(Spec); 

sumAk = sum(Ak(:,2:end-1),2);
fk = (1:N/2-1)*sr/N;

% -- Spectral moments

%% Spectral centroid
Sc = sum(Ak(:,2:end-1).*(ones(size(Ak,1),1)*fk),2)./sumAk;

%% 2nd order moment
M2 = sum(Ak(:,2:end-1).*(ones(size(Ak,1),1)*(fk.^2)),2)./sumAk;
%% 3rd order moment
M3 = sum(Ak(:,2:end-1).*(ones(size(Ak,1),1)*(fk.^3)),2)./sumAk;
%% 4th order moment
M4 = sum(Ak(:,2:end-1).*(ones(size(Ak,1),1)*(fk.^4)),2)./sumAk;

%% Spectral width from M2
Sw = (M2-Sc.^2).^0.5;

%% Spectral asymmetry from M3
Sa = (2*Sc.^3-3*Sc.*M2+M3)./(Sw.^3);

%% Spectral flatness from M4
Sf = (-3*Sc.^4+6*Sc.*M2-4*Sc.*M3+M4)./(Sw.^4)-3;

Sx=[Sc Sw Sa Sf];
sx=mean(Sx,1);
%endfunction

function so=features_specOsc(Spec)
% specOsc - compute spectral oscillation, after [Peeters, 04]
%
% SYNTAX
% So=specOsc(Spec)
%
% ARGUMENTS
% Spec - "Framed spectrum matrix" i.e a matrix
%        where each row is the spectrum of a signal frame
%
% OUTPUT
% so - Spectral oscillation

A=abs(Spec);
So=mGeomean(A')'./mean(A,2);
so=mean(So,1);
%endfunction

function ss=features_specSlope(Spec,N,sr)
% specSlope - compute spectral slope, after [Peeters, 04]
%
% SYNTAX
% Ss=specSlope(Spec)
%
% ARGUMENTS
% Spec - "Framed spectrum matrix" i.e a matrix
%        where each row is the spectrum of a signal frame
%
% OUTPUT
% So - Spectral slope
% N  - frame length (size(xn,2))
% sr - sampling freq

A=abs(Spec);
freqs=(0:N/2)*sr/N;
R=N/2; % hopsize

Ss= (R*sum(repmat(freqs,size(A,1),1).*A,2)-repmat(sum(freqs,2),size(A,1),1).*sum(A,2))./...
      (R*sum(repmat(freqs.^2,size(A,1),1),2)-(sum(repmat(freqs,size(A,1),1),2)).^2);
ss=mean(Ss,1);
%endfunction

function ar=features_lpcCoef(Frames)
% Computing first AR coefs  
  lenf=2;  
  Ar=lpc(Frames',lenf);
  Ar=Ar(:,2:end);
  ar=mean(Ar,1);
%endfunction

function mfcc=features_mfcc(Frames,Spec,N,sr)

% ----- !!!! A COMPLETER !!!! ------ %

%CP=melcepst(...
  
mfcc=mean(CP,1);
  
%enfunction  

function fc = features_freqCutoff(Spec,sr,N)
% freqCutoff - Compute frequency  cutoff (in Hz)
%               (freq below which 99% of spectrum power has been 
%                accounted for)
%
% SYNTAX
% fc = freqCutoff(Spec,sr,N)
%
% ARGUMENTS
% Spec - "Framed spectrum matrix" i.e a matrix
%        where each row is the spectrum of a signal frame
% N  - frame length (size(xn,2))
% sr - sampling freq
%
% OUTPUT
% fc - cutoff frequencies

  
  % ----- !!!! A COMPLETER !!!! ------ %
  
%endfunction
  

  
% Standardise un jeu de donn?es
function Y = standardize(X)
  
% ----- !!!! A COMPLETER !!!! ------ %

%endfunction


% Mod?le gaussien, matrice de covariance diagonale
% (pas assez de donn?es pour estimer une matrice de covariance 20x20!
function model = gaussian_train(training_set, training_classes)
  Nclasses = max(training_classes);
  Ndata = size(training_set, 2);
  
  mu = zeros(Nclasses, Ndata);
  sigma2 = mu;
  
  for i=1:Nclasses,
    data = training_set(find(training_classes == i),:);
    mu(i,:) = mean(data,1);
    sigma2(i,:) = var(data,1);
  end;
  model = struct('mu', mu, 'sigma2', sigma2);
%endfunction

function class = gaussian_classify(vector, model)
  mu = model.mu;
  sigma2 = model.sigma2;
  Nclasses = size(mu, 1);
  ll = ones(Nclasses, 1) * vector;
  ll = ll - mu;
  ll = ll .* ll;
  ll = ll ./ sigma2;
  ll = - 1/2 * (sum(ll, 2)  + sum(log(sigma2),2) + size(mu,2) * log(2 * pi));
  [minv, minindex] = max(ll);
  class = minindex;
%endfunction


% Classification par l'algorithme du plus proche voisin
function model = knn_train(training_set, training_classes)
  model = struct('X', training_set, 'Y', training_classes);
%endfunction

function class = knn_classify(vector, model)
  X = model.X; % vecteurs d'attributs des morceaux de la base d'apprentissage (54 x 20)
  Y = model.Y; % instruments des morceaux de la base d'apprentissage (54x1)
  
  % ----- !!!! A COMPLETER !!!! ------ %
  
%endfunction

function test_piccolo(database)
  database.filenames{61,1} = 'piccolo.wav';
  database.features = compute_features(database,61);
  database.nfeatures = standardize(database.features(:,:));
  model = knn_train(database.nfeatures(1:60,:), database.instruments);
  fprintf('1-NN : %d\n', knn_classify(database.nfeatures(61,:), model));
  model = gaussian_train(database.nfeatures(1:60,:), database.instruments);
  fprintf('gaussian model : %d\n', gaussian_classify(database.nfeatures(61,:), model));
%endfunction

function db = cross_validation(database, train_fun, classify_fun)
  folds = 10;
  rand('seed', 11);
  bins = floor(linspace(1,folds+1-1e-10, length(database.instruments)));
  [foo, shuffle] = sort(rand(1, length(bins)));
  bins = bins(shuffle);
  for fold = 1:folds,
    training = find(bins ~= fold);
    testing = find(bins == fold);
    model = feval(train_fun,database.nfeatures(training,:), database.instruments(training));
    for i=1:length(testing),
      database.classifications(testing(i)) = feval(classify_fun,database.nfeatures(testing(i),:), model);
    end;
  end;
  db = database;
%endfunction

% Affiche les r?sultats
function display_results(database)
  groundtruth = database.instruments;
  results = database.classifications;
  confusion = zeros(max(groundtruth), max(groundtruth)+1);
  for i=1:length(groundtruth),
    if results(i) > 0
      confusion(groundtruth(i), results(i)) = confusion(groundtruth(i), results(i)) + 1;
   end;
  end;
  correct = sum(diag(confusion));
  fprintf('Correctly classified: %f %%\nConfusion:\n', 100*correct / length(groundtruth));
  disp(confusion)
%endfunction
