function [rxcode, cnumerr] = ReedSolomonRecon(msg,corr_msg,m,n,k)
%ReedSolomonRecon Input: original matrix, one message per row; noisy
%message (without the parity code), one message per row; (m) cardinality of 
%the symbol; (n) lenggth of the message; (k) length of the information
% Output: recovered message
msg = double(msg); corr_msg = double(corr_msg);m = double(m); n =double(n); k = double(k);
msg = gf(msg,m);
code = rsenc(msg,n,k);
corr_msg = [corr_msg code(:,k+1:end)];
[rxcode, cnumerr] = rsdec(corr_msg, n,k);
rxcode(cnumerr<0,:) = corr_msg(cnumerr<0,1:k);
rxcode = rxcode.x;
% cnumerr
end