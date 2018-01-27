function [out1, out2] = KMM_PSConS_obj_withLambdaSP(params, R, sigma, cal_vec, x, y, xtst, lambda_SP)
%calculate the objective function (kernel mean matching for
%shape-preserving conditional shift) and its derivatives
%   Inputs:
%       params: the vector of w_q and b_q;
%       cal_vec: the vector (L+\lambda I)^{-1}*L * 1

[m,d] = size(x);
n = size(xtst,1);
length_G = length(params)/2;
% lambda_SP = .001 /m * m^2; % for regularization...

% construct x_new
Col_R = size(R,2);
W = R * vec2mat(params(1:length(params)/2), Col_R)';
B = R * vec2mat(params(1+length(params)/2:end), Col_R)';

% x_new = x * diag(params(1:d)) + repmat(params(d+1:end)', m, 1);
x_new = x .* W + B;

% the kernel matrices
tilde_K = rbf_dot(x_new,x_new,sigma,0);

tilde_Kc = rbf_dot(xtst, x_new, sigma, 0);

% objective function
out1 = cal_vec' * tilde_K * cal_vec - 2*m/n * ones(1,n) * tilde_Kc * cal_vec + ...
    lambda_SP * ( sum(sum( (W-ones(size(W))).^2)) + sum(sum(B.^2)) );

% its partial derivatives
if nargout == 2               % ... and if requested, its partial derivatives
    out2 = zeros(size(params));       % set the size of the derivative vector
    for i = 1:length_G
        pp = mod(i-1,Col_R) + 1; % to calculate the gradient w.r.t. G_{pp,qq} and H_{pp,qq}.
        qq = floor((i-1)/Col_R)+1; % max is Dim
        
        tmp1 = repmat(x_new(:,qq)', n, 1) - repmat(xtst(:,qq), 1, m);
        
        tmp2 = repmat(x_new(:,qq)', m, 1) - repmat(x_new(:,qq), 1, m);
        
        tilde_E_pq = -tmp1  .* repmat(R(:,pp)', n, 1);
        
        E_pq = tilde_E_pq .* repmat(x(:,qq)', n, 1);
        
        tilde_D_pq = -tmp2 .* ( repmat(R(:,pp)', m, 1) - repmat(R(:,pp), 1, m) );
        
        D_pq = -tmp2 .* ( repmat((x(:,qq) .* R(:,pp))', m, 1) - repmat(x(:,qq) .* R(:,pp), 1, m) );
        
        out2(i+length_G) = cal_vec' * (tilde_D_pq .* tilde_K ) * cal_vec /sigma^2 -...
            2*m/n * ones(1,n) * (tilde_E_pq .* tilde_Kc) * cal_vec /sigma^2; % for b
        % for w
        out2(i) = cal_vec' * ( D_pq.* tilde_K ) * cal_vec /sigma^2 -...
            2*m/n * ones(1,n) * ( E_pq  .* tilde_Kc) * cal_vec/sigma^2;
    end
    
    % the penalty term
    out2(1:length_G) = out2(1:length_G) + reshape(2*lambda_SP * R' * (W-ones(size(W))), length_G,1);
    out2(length_G+1:end) = out2(length_G+1:end) + reshape(2*lambda_SP * R' * B, length_G,1);
    
    %         for i = 1:d
    %         Fq = - ( repmat(x_new(:,i)', n, 1) - repmat(xtst(:,i), 1, m)); % to save computations, some scale product will be performed later
    %         out2(i+d) = - 2*m/n * ones(1,n) * (Fq .* tilde_Kc) * cal_vec /sigma^2; % for b
    %         % for w
    %         out2(i) = cal_vec' * ( sq_dist(x(:,i)').* tilde_K ) * cal_vec * (-params(i)/sigma^2) -...
    %             2*m/n * ones(1,n) * ( (repmat(x(:,i)', n, 1).* Fq)  .* tilde_Kc) * cal_vec/sigma^2;
    %     end
    
end

end