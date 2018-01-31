function [beta] = betaKMM_targetshift(X, Y, Xtst, Ytst, sigma, width_L_beta, lambda_beta);
% function [beta EXITFLAG] = betaKMM_targetshift(X, Y, Xtst, Ytst, sigma, width_L_beta, lambda_beta);
% This function estimates the weight \beta for correcting for target
% shift.
% Input:
%       X: features on training domain (size: # training data points * # dimensions);
%       Y: target on training domain (size: # training data points * 1);
%       X: features on test domain (size: # test data points * # dimensions);
%       Y: target on training domain (size: # tset data points * 1);
%       sigma: the kernel width for X used to construct Gram matrix K;
%       width_L_beta & lambda_beta: the kernel width and regularization 
%       parameter lambda_beta used for L_beta for continuous Y (regression).
% Outut:
%       beta: estimated weights (size: # training data points * # dimensions);


nsamples = size(X,1);  % number of train samples
ntestsamples = size(Xtst,1);  % number of test samples

Centering_x = 0;
Centering_Ky = 0;
Scheme = 2; % 2 % now specified as a parameter
% options = optimset('quadprog');
options = optimset('MaxIter', 3000); % 5000
Thresh_beta = 1E-3; %2E-1; % 1E-3
Thresh_discrete = 16;

dd = size(X,2);
if ~exist('sigma', 'var')|isempty(sigma)
if nsamples < 500
sigma = 0.4 * sqrt(dd); %1E-1; % 1E-3
elseif nsamples < 1000
sigma = 0.2 * sqrt(dd);
else
sigma = 0.14 * sqrt(dd);
end

% minimize...
'calculating H=K...'
if size(X,2) > 1E4
    mean_std_x = mean(std(X(:,1:10:end)));
else
    mean_std_x = mean(std(X));
end

sigma = sigma*mean_std_x

end


% variables: (here in the program / in (12) in the paper)
% H is K
% f is kappa
%

H = rbf_dot(X,X,sigma,0);
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

% to save time, we only run GPR once; later just load the values of
% logtheta_y, KY_x, and pdinv_KY_x
% global WhichDataSet;
% if WhichDataSet <= 0
    
    logtheta0 = [log(width)*ones(size(Y,2),1); 0; log(sqrt(0.1))];
    fprintf('Optimizing hyperparameters in GP regression:\n');
    %     [logtheta_x, fvals_x, iter_x] = minimize(logtheta0, 'gpr_multi', -150, covfunc, z, 1/std(eix(:,1)) * eix);
    %     [logtheta_y, fvals_y, iter_y] = minimize(logtheta0, 'gpr_multi', -150, covfunc, z, 1/std(eiy(:,1)) * eiy);
    % -200 or -350?
    
    %old gpml-toolbox
    %
    IIx = find(eig_Kx > max(eig_Kx) * Thresh); eig_Kx = eig_Kx(IIx); eix = eix(:,IIx);
    
    Xtmp = 2*sqrt(nsamples) *eix * diag(sqrt(eig_Kx))/sqrt(eig_Kx(1));
    GP_sample_ratio = ceil(length(Y) / 2500);
    [logtheta_y, fvals_y, iter_y] = minimize(logtheta0, 'gpr_multi', -350, covfunc, Y(1:GP_sample_ratio:end), Xtmp(1:GP_sample_ratio:end, :));

%     [logtheta_y, fvals_y, iter_y] = minimize(logtheta0, 'gpr_multi', -350, covfunc, Y, 2*sqrt(nsamples) *eix * diag(sqrt(eig_Kx))/sqrt(eig_Kx(1)));
    
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
    
    L_inv_L = KY_x*pdinv(KY_x + exp(2*logtheta_y(end))*eye(nsamples));
    
%     if (WhichDataSet == 0)
%         WhichDataSet = floor(cputime);
%         fname = sprintf('Hyper_Y_GPR_dataset%i.mat', WhichDataSet);
%         save(fname, 'logtheta_y','KY_x','L_inv_L');
%     end
% else
%     fname = sprintf('Hyper_Y_GPR_dataset%i.mat', WhichDataSet);
%     load(fname);
% end

J = L_inv_L * H * L_inv_L'; J = (J+J')/2;

%%%


'calculating f=kappa...'
R3 = rbf_dot(X,Xtst,sigma,0);
M= L_inv_L * (R3*ones(ntestsamples, 1));
M=-nsamples/ntestsamples*M;

% did the same, but slowlier:
% f=-nsamples/ntestsamples*ones(nsamples,1);
% for i=1:nsamples
%     fi=0;
%     for j=1:ntestsamples
%         fi = fi + rbf_dot(X(i,:),Xtst(j,:),sigma);
%     end
%     f(i,1) = f(i,1)*fi;
% end
%
% do they really the same?
%'different f?'
%[f1 f]

% subject to...
% abs(sum(beta_i) - m) <= m*eps
% which is equivalent to A*beta <= b where A=[1,...1;-1,...,-1] and b=[m*(eps+1);m*(eps-1)]


if Scheme == 1 || Scheme == 3
    eps = (sqrt(nsamples)-1)/sqrt(nsamples);
    %eps=1000/sqrt(nsamples);
    A=ones(1,nsamples);
    A(2,:)=-ones(1,nsamples);
    b=[nsamples*(eps+1); nsamples*(eps-1)];
    
    Aeq = [];
    beq = [];
    % 0 <= beta_i <= 1000 for all i
    LB = zeros(nsamples,1)*Thresh_beta;
    UB = ones(nsamples,1).*100; % 1000
    
    % X=QUADPROG(H,f,A,b,Aeq,beq,LB,UB) attempts to solve the quadratic programming problem:
    %              min 0.5*x'*H*x + f'*x
    % subject to:  A*x <= b
    %              Aeq*x = beq
    %              LB <= x <= UB
    
    'solving quadprog for betas...'
    [beta_t,FVAL,EXITFLAG] = quadprog_oct(J,M,A,b,Aeq,beq,LB,UB,[],options);
    EXITFLAG,
    if Scheme == 1
        beta = max(beta_t, Thresh_beta);
    else
        beta{2} = max(beta_t, Thresh_beta);
    end
    %     if ((EXITFLAG==0 ) && (doingRealTraining==1))
    %         %[beta,FVAL,EXITFLAG] = quadprog(H,f,A,b,Aeq,beq,LB,UB,beta,optimset('MaxIter',1e4));
    %         EXITFLAG
    %     end
    %
    %     if (regression==0)
    %         % guarantee that all beta greater than 0
    %         threshold=0.01*abs(median(beta));
    %         beta(beta<threshold) = threshold;
    %         sprintf('number of beta < %f: %d (0 is good)', threshold, length(find(beta<threshold)))
    %     end
end

if Scheme == 2 || Scheme == 3
    % is Y discrete or continous ?
    UB_beta = 10; % the maximum value of beta
    % calculate the cardinality of Y and its possible values:
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
    if Card_Y < Thresh_discrete+1
        fprintf('Y is discrete; calculateing beta is easier...\n');
        R = zeros(nsamples, Card_Y);
        for i = 1:nsamples
            R(i,Y(i) == Val_Y)  = 1;
        end
        % lb_sum_beta = 0.95; ub_sum_beta = 1.05;
        B = UB_beta; lb_sum_beta = 1-B/sqrt(nsamples)/4; ub_sum_beta =  1+B/sqrt(nsamples)/4;
        A = [-R; R; ones(1,nsamples)*R; -ones(1,nsamples)*R];
        b = [zeros(nsamples, 1)-Thresh_beta; ones(nsamples,1)*UB_beta;ub_sum_beta*nsamples; -lb_sum_beta*nsamples]; % 0.02
        [alpha, FVAL, EXITFLAG] = quadprog(R'*J*R, R'*M, A, b,...
            [], [], zeros(Card_Y,1), 100*ones(Card_Y,1),ones(Card_Y,1), options);
        if Scheme == 2
            beta = max(R * alpha, Thresh_beta);
        else
            beta{1} = max(R * alpha, Thresh_beta);
        end
    else
        fprintf('Y is continuous; please wait a while for beta values...\n');
        lambda2 = 0.1; %0.1
        % re-calculate the kernel matrix on Y?
        %     KY = (KY_x + KY_x')/2;
        %         Width_KY = width*mean(std(Y)) * 3; %2
        Width_KY = width_L_beta;
        lambda2 = lambda_beta;
        KY = rbf_dot(Y,Y,Width_KY,0);
        KY = (KY + KY')/2;
        
        if nsamples < 600
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
            alpha0 = pdinv(Q'*Q) * Q'* ones(nsamples,1);
        end
        
        
        
        %     [alpha, FVAL, EXITFLAG] = quadprog(Q'*J*Q, Q'*M, -Q, zeros(nsamples, 1),...
        %         ones(1,nsamples)*Q, nsamples, -1000*ones(nsamples,1), 1000*ones(nsamples,1),[], options);
        
        %         lb_sum_beta = 0.8; ub_sum_beta = 1.2;
        B = 10; lb_sum_beta = 1-B/sqrt(nsamples)/4; ub_sum_beta =  1+B/sqrt(nsamples)/4;
        A = [-Q; Q; ones(1,nsamples)*Q; -ones(1,nsamples)*Q];
        b = [zeros(nsamples, 1)-Thresh_beta; ones(nsamples,1)*UB_beta; ub_sum_beta*nsamples; -lb_sum_beta*nsamples]; % 0.02
        [alpha, FVAL, EXITFLAG] = quadprog(Q'*J*Q, Q'*M, A, b,...
            [], [], -1E4*ones(size(Q,2),1), 1E4*ones(size(Q,2),1),alpha0, options);
        if Scheme == 2
            beta = max(Q * alpha, Thresh_beta);
        else
            beta{1} = max(Q * alpha, Thresh_beta);
        end
    end
end
