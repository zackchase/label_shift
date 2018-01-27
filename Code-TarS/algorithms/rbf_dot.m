%Radial basis function inner product
%Arthur Gretton

%Pattern input format : [pattern1 ; pattern2 ; ...]
%Output : Matrix of RBF values k(x1,x2)
%Deg is kernel size

% modified to reduce memory usage
% Zhikun Wang

function [H]=rbf_dot(patterns1,patterns2,deg, flag);

% assert(false, 'function abadaned... use sq_dist instead');
%Note : patterns are transposed for compatibility with C code.

size1=size(patterns1);
size2=size(patterns2);


G = sum((patterns1.*patterns1),2);
H = sum((patterns2.*patterns2),2);

H = repmat(G,1,size2(1)) + repmat(H',size1(1),1) - 2*patterns1*patterns2';

H=exp(-H/2/deg^2);

