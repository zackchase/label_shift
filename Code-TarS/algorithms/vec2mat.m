function [mat, err] = vec2mat(vec, m, ad_to);
%VEC2MAT Vector to matrix conversion.
%       MAT = VEC2MAT(VEC, M) converts VEC to be a M column matrix with row
%       priority arrangement. If it needs to add an element, 0 will be added
%       to the end of the input vector to form the required matrix.
%
%       MAT = VEC2MAT(VEC, M, AD_TO) specifies the adding element to be the
%       value in AD_TO.
%
%       [MAT, ERR] = VEC2MAT(...) outputs ERR, which specifies how many
%       elements has been added in constructing the matrix.
%       Wes Wang 7/12/95, 10/11/95.
%       Copyright (c) 1995-96 by The MathWorks, Inc.
%       $Revision: 1.1 $  $Date: 1996/04/01 18:04:27 $
if nargin < 2
    error('Not enough input variable.');
elseif nargin < 3
    ad_to = 0;
end;
% make vec to be a column vector.
[n_vec, m_vec] = size(vec);
if m_vec == m
    mat = vec;
    return; % nothing to do.
elseif n_vec > 1
    vec = vec';
    vec = vec(:)';
    [n_vec, m_vec] = size(vec);
end;
LC = ceil(m_vec / m); % number of elements.
err = LC * m - m_vec; % number of element to be added.
mat = zeros(m, LC);
len_ad = length(ad_to);
if len_ad == 1
    mat(:) = [vec, zeros(1, err) + ad_to];
elseif len_ad >= err
    ad_to = ad_to(:)';
    mat(:) = [vec, ad_to(1:err)];
else
    mat(:) = [vec, ad_to, zeros(1, err-len_ad) + ad_to(len_ad)];
end
mat = mat';
%%--end of VEC2MAT--