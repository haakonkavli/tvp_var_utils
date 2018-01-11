function [Xlag] = lag_matrix(X,p)

% lag_matrix: Summary of this function goes here
%   Detailed explanation goes here
[Traw,N]=size(X);
Xlag=zeros(Traw,N*p);
for ii=1:p
    Xlag(p+1:Traw,(N*(ii-1)+1):N*ii)=X(p+1-ii:Traw-ii,1:N);
end

