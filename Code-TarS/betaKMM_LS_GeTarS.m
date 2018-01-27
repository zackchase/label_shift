function [beta, params, X_new, W, B] = betaKMM_LS_GeTarS(X, Y, Xtst, Ytst, sigma, lambda_SP, greedy_ratio);
% function [beta, params, X_new, W, B] = betaKMM_LS_GeTarS(X, Y,...
%                                   Xtst, Ytst, sigma, lambda_SP, greedy_ratio);
% This function estimates the weights beta and transformed training data for
%   correcting for Location-Scale Generalized Target Shift (LS-GeTarS).
% Input:
%       X: features on training domain (size: # training data points * # dimensions);
%       Y: target on training domain (size: # training data points * 1);
%       X: features on test domain (size: # test data points * # dimensions);
%       Y: target on training domain (size: # tset data points * 1);
%       sigma: the kernel width for X used to construct Gram matrix K;
%       lambda_SP: the regularization parameter lambda_LS in the paper
%              (default value: 1E-3 * # training data points);
%       greedy_ratio: a trade-off parameter for the alternate optimization
%       procedure to avoid beging too greedy (default value: 0.9).
% Outut:
%       beta: estimated weights (size: # training data points * # dimensions);
%       params: estimated parameters for W and B;
%       X_new: transformed training points (of the same size as X);
%       W and B: the LS transformation parameters.

[nsamples, Dim] = size(X);  % number of train samples
ntestsamples = size(Xtst,1);  % number of test samples
beta_oracle = ones(nsamples,1);
Thresh_beta = 1E-3; %2E-1; % 1E-3
options = optimset('quadprog');
options = optimset('MaxIter', 3000); % 3000
Tol = 1E-2;
Max_Iter = 20;
Thresh_beta = 1E-3;
Thresh_discrete = 16;
Avoid_greedy = 1;
Max_nsamples_noDR = 400;

UB_beta = 10; % upper bound of beta
lambda1 = 0.1; %0.1

% parameters
Centering_x = 0;
Centering_Ky = 0;
width_L_beta = 3*sigma;
lambda_beta = 0.1;

if ~exist('lambda_SP', 'var')|isempty(lambda_SP)
    lambda_SP = 1E-3; %1E-1; % 1E-3
end
lambda_SP = lambda_SP / nsamples * nsamples^2;

if ~exist('greedy_ratio', 'var')|isempty(greedy_ratio)
    greedy_ratio = 0.8;
end


% fprintf('calculating A...\n')
if size(X,2) > 1E4
    mean_std_x = mean(std(X(:,1:10:end)));
else
    mean_std_x = mean(std(X));
end
Sigma = sigma*mean_std_x;

% kernel matrix
H = rbf_dot(X,X,Sigma,0);
% H=(H+H')/2; %make the matrix symmetric (it isn't symmetric before because of bad precision)

%%% learn the kernel matrix of Y and the regularization parameter
Thresh = 1E-5;
Hc = eye(nsamples) - ones(nsamples,nsamples)/nsamples;
if Centering_x
    H_new = Hc * H * Hc;
else
    H_new = H;
end
H_new = (H_new + H_new')/2;
if issparse(H_new)
    [eix, eig_Kx] = eigs(H_new, min(400, floor(nsamples/4))); % /2
else
    [eig_Kx, eix] = eigdec(H_new, min(400, floor(nsamples/4))); % /2
end
% disp('  covfunc = {''covSum'', {''covSEard'',''covNoise''}};')
covfunc = {'covSum', {'covSEard','covNoise'}};
if nsamples < 200
    width = 0.8;
elseif nsamples < 1200
    width = 0.5;
    %        width = 0.8;
else
    width = 0.3; % 0.3
end

logtheta0 = [log(width)*ones(size(Y,2),1); 0; log(sqrt(0.1))];
fprintf('Optimizing hyperparameters in GP regression:\n');
%     [logtheta_x, fvals_x, iter_x] = minimize(logtheta0, 'gpr_multi', -150, covfunc, z, 1/std(eix(:,1)) * eix);
%     [logtheta_y, fvals_y, iter_y] = minimize(logtheta0, 'gpr_multi', -150, covfunc, z, 1/std(eiy(:,1)) * eiy);
% -200 or -350?

%old gpml-toolbox
%
IIx = find(eig_Kx > max(eig_Kx) * Thresh); eig_Kx = eig_Kx(IIx); eix = eix(:,IIx);
[logtheta_y, fvals_y, iter_y] = minimize(logtheta0, 'gpr_multi', -350, covfunc, Y, 2*sqrt(nsamples) *eix * diag(sqrt(eig_Kx))/sqrt(eig_Kx(1)));

covfunc_Y = {'covSEard'};
KY_x = feval(covfunc_Y{:}, logtheta_y, Y);

% Note: in the conditional case, no need to do centering, as the regression
% will automatically enforce that.

% Kernel matrices of the errors
if Centering_Ky
    Hc1 = eye(nsamples) - ones(nsamples,nsamples)/nsamples;
    %     KY_x = Hc1 * KY_x * Hc1; %%%??? or
    KY_x = Hc1 * KY_x * Hc1;
end

% will be used in the iterations
pdinv_KY_x = pdinv(KY_x + exp(2*logtheta_y(end))*eye(nsamples));
L_inv_L = KY_x*pdinv_KY_x;


%% finding Q which transforms the parameters to W and B
% Y discrete or continuous ?
Card_Y = 1;
Val_Y = [];
Val_Y(1) = Y(1);
for i=2:nsamples
    if ~max(Y(i)==Val_Y)
        Val_Y = [Val_Y Y(i)];
        Card_Y = Card_Y + 1;
        if Card_Y > Thresh_discrete % continuous, so no need to calculate the cardinality.
            break;
        end
    end
end
if Card_Y < Thresh_discrete + 1
    fprintf('Y is discrete; fewer parameters...\n');
    % for beta
    R = zeros(nsamples, Card_Y);
    for i = 1:nsamples
        R(i,Y(i) == Val_Y)  = 1;
    end
    % lb_sum_beta = 0.95; ub_sum_beta = 1.05;
    B = UB_beta; lb_sum_beta = 1-B/sqrt(nsamples)/4; ub_sum_beta =  1+B/sqrt(nsamples)/4;
    A_beta = [-R; R; ones(1,nsamples)*R; -ones(1,nsamples)*R];
    b_beta = [zeros(nsamples, 1)-Thresh_beta; ones(nsamples,1)*UB_beta;ub_sum_beta*nsamples; -lb_sum_beta*nsamples]; % 0.02
    
    % initialization...
    beta0 = ones(nsamples,1);
    alpha_beta0 = ones(Card_Y,1);
    
    % for SP_ConS
    Q = R;
    Col_Q = Card_Y;
    params0 = [ones(Dim*Col_Q,1); zeros(Dim*Col_Q,1)]; % for [W ; B]
    
else
    fprintf('Y is continuous; more parameters needed...\n');
    Width_KY = width_L_beta;
    lambda2 = lambda_beta;
    KY = rbf_dot(Y,Y,Width_KY,0);
    KY = (KY + KY')/2;
    
    if nsamples < Max_nsamples_noDR
        Q = KY * pdinv(KY + lambda2 * eye(nsamples));
        alpha0 = ones(nsamples,1);
    else
        pdinv_KY = pdinv(KY + lambda2 * eye(nsamples));
        %         [eig1, ei1] = eigdec(KY * pdinv_KY * pdinv_KY'*KY', min(300, floor(nsamples/4))); % /2
        %         II1 = find(eig1 > max(eig1) * Thresh); eig1 = eig1(II1); ei1 = ei1(:,II1);
        %         R = ei1 * diag(sqrt(eig1));
        [UU1,SS1,VV1] = svd(KY *pdinv_KY);
        eig1 = diag(SS1);
        II1 = find(eig1 > max(eig1) * Thresh);
        Q = KY*pdinv_KY * VV1(:,II1);
        tmp0 = pdinv(Q'*Q) * Q'* ones(nsamples,1);
        Col_Q = size(Q,2);
        params0 = [repmat(tmp0, Dim, 1); zeros(Dim*Col_Q,1)];
    end
    
    % for beta
    R = Q;
    %         lb_sum_beta = 0.8; ub_sum_beta = 1.2;
    B = UB_beta; lb_sum_beta = 1-B/sqrt(nsamples)/4; ub_sum_beta =  1+B/sqrt(nsamples)/4;
    A_beta = [-R; R; ones(1,nsamples)*R; -ones(1,nsamples)*R];
    b_beta = [zeros(nsamples, 1)-Thresh_beta; ones(nsamples,1)*UB_beta; ub_sum_beta*nsamples; -lb_sum_beta*nsamples]; % 0.02
    
    % initialization...
    beta0 = ones(nsamples,1);
    if nsamples < Max_nsamples_noDR
        alpha_beta0 = (KY + lambda2*eye(nsamples)) * pdinv((KY + 1E-5*std(Y)*eye(nsamples))) * beta0;
    else
        alpha_beta0 = pdinv(R'*R) * R'* beta0;
    end
    
end

%

%% alternate optimization

Error = 1;
Iter = 0;
tilde_K = H;
tilde_Kc = rbf_dot(Xtst,X,Sigma,0);
while Error > Tol & Iter < Max_Iter
    Iter = Iter + 1,
    
    % over beta
    % construct J_beta and M_beta
    J_beta = L_inv_L * tilde_K * L_inv_L'; J_beta = (J_beta+J_beta')/2;
    M_beta= L_inv_L * (tilde_Kc'*ones(ntestsamples, 1));
    
    M_beta=-nsamples/ntestsamples*M_beta;
    % first optimizing over beta
    
    if Card_Y < Thresh_discrete + 1
        [alpha_beta, FVAL_beta, EXITFLAG_beta] = quadprog(R'*J_beta*R, R'*M_beta, A_beta, b_beta,...
            [], [], zeros(Card_Y,1), 100*ones(Card_Y,1),alpha_beta0, options);
    else
        [alpha_beta, FVAL_beta, EXITFLAG_beta] = quadprog(R'*J_beta*R, R'*M_beta, A_beta, b_beta,...
            [], [], -1E4*ones(size(R,2),1), 1E4*ones(size(R,2),1),alpha_beta0, options);
    end
    %%% to avoid "too greedy" solutions... Let beta go back for a while...
    if Avoid_greedy
        alpha_beta = greedy_ratio*alpha_beta + (1-greedy_ratio)*alpha_beta0;
        beta = greedy_ratio*R*alpha_beta + (1-greedy_ratio)*beta0;
    end
    %%%
    
    alpha_beta0 = alpha_beta;
    beta = max(R * alpha_beta, Thresh_beta);
    
    
    % over W and B
    params = minimize(params0, 'KMM_PSConS_obj_withLambdaSP', -40, Q, Sigma, L_inv_L'*beta, X, Y, Xtst,lambda_SP);
    
    greedy_ratio2 = greedy_ratio; %%%
    if Avoid_greedy
        params = greedy_ratio2*params + (1-greedy_ratio2)*params0;
    end
    
    
    Error = max(sqrt(sum((beta-beta0).^2)/nsamples), sqrt(sum((params-params0).^2)/nsamples)),
    beta0 = beta;
    params0 = params;
    
    % calculate tilde_K and tilde_Kc
    % will be used in the next iteration
    W = Q * vec2mat(params(1:length(params)/2), Col_Q)';
    B = Q * vec2mat(params(1+length(params)/2:end), Col_Q)';
    X_new = X .* W + B;
    tilde_K = rbf_dot(X_new,X_new,Sigma,0);
    tilde_Kc = rbf_dot(Xtst, X_new, Sigma, 0);
    
end
