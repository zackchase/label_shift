function [beta] = betaEM_targetshift(X, Y, Xtst, Ytst);
% function [beta] = betaEM_targetshift(X, Y, Xtst, Ytst);
% EM to find to find the change in the prior distribution of Y for
% classification proposed by Chan & Ng (2005): Word sense disambiguation
% with distribution estimation. In Proceedings. IJCAI 2005.
% Netlab toolbox (http://www1.aston.ac.uk/eas/research/groups/ncrg/resources/netlab/) 
% is needed to run this function.

nsamples = size(X,1);  % number of train samples
ntestsamples = size(Xtst,1);  % number of test samples

Ind_0 = find(Y==0); 
Ind_1 = find(Y==1);
P_tr = zeros(nsamples, 1);
P_tr(Ind_0) = length(Ind_0)/nsamples; 
P_tr(Ind_1) = 1-length(Ind_0)/nsamples;

mu(1,:) = mean(X(Ind_0,:));
mu(2,:) = mean(X(Ind_1,:));
Sigma(:,:,1) = (X(Ind_0,:) - repmat(mu(1,:), length(Ind_0), 1))' *...
    (X(Ind_0,:) - repmat(mu(1,:), length(Ind_0), 1)) /length(Ind_0);
Sigma(:,:,2) = (X(Ind_1,:) - repmat(mu(2,:), length(Ind_1), 1))' *...
    (X(Ind_1,:) - repmat(mu(2,:), length(Ind_1), 1)) /length(Ind_1);

% iteration
Error = 1;
Tol = 1E-4;

% create the structure
mix = gmm(2, 2, 'full');
mix.priors = [P_tr(Ind_0(1)) P_tr(Ind_1(1))];
mix.centres(1,:) = mu(1,:);
mix.centres(2,:) = mu(2,:);
mix.covars = Sigma;
init_priors = mix.priors;

while Error > Tol
      [post, act] = gmmpost(mix, Xtst);
      new_pr = sum(post, 1);
      mix.priors = new_pr ./ ntestsamples;
      Error = norm(mix.priors - init_priors);
      init_priors = mix.priors;
end

P_te = zeros(nsamples,1);
P_te(Ind_0) = mix.priors(1); 
P_te(Ind_1) = mix.priors(2);

beta = P_te ./ P_tr;
end