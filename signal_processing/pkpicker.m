function  [peaks, locs] = pkpicker( x, thresh, number, sortem )

% Pick out the peaks in a vector
%
% [peaks,locs] = pkpicker(x, thresh, number, sortem)
%
% peaks  : peak values
% locs   : location of peaks (index within a column)
% x      : input data  (if complex, operate on mag)
% thresh : reject peaks below this level
% number : max number of peaks to return
% sortem : 'sort' will return list sorted by peak height,
%           instead of by index # (the default)
%
% Example:
%
% Nrec=250
% x=-diff(normpdf([-1:.01:1],0,.1),2); xnorm=x/max(abs(x));
% xper=ones(1,5); xper=upsample(xper,Nrec); xper=conv(xper,xnorm);
% thresh=0.5; number=Inf;
% [peaks,locs] = pkpicker(xper, thresh, number);
% Nrec_vect=diff(locs)'
% peaks=peaks'
% locs=locs'
% figure; plot(xper); hold on; stem(locs,peaks,'-or'); 
% grid; ylim([-0.5 1.2]); xlabel('# sample'); ylabel('Amplitude')
% legend ('Original waveform','Detected peaks')

%---------------------------------------------------------------
% copyright 1994, by C.S. Burrus, J.H. McClellan, A.V. Oppenheim,
% T.W. Parks, R.W. Schafer, & H.W. Schussler.  For use with the book
% "Computer-Based Exercises for Signal Processing Using MATLAB"
% (Prentice-Hall, 1994).
%---------------------------------------------------------------

if nargin == 1
   thresh = -inf;   number = -1;   sortem = 0;
elseif nargin == 2
   number = -1;   sortem = 0;
elseif nargin == 3
   sortem = 0;
end
if strcmp(sortem,'sort')
   sortem = 1;
end

[M,N] = size(x);
if M==1
   x = x(:);     %-- make it a single column
   [M,N] = size(x);
end
if any(imag(x(:))~=0)
   x = abs(x);       %---- complex data, so work with magnitude
end
for kk = 1:N
  mask = diff( sign( diff( [x(1,N)-1;  x(:,N); x(M,N)-1] ) ) );
  jkl = find( mask < 0 & x >= thresh);
  if (number>0 & length(jkl)>number)
     [tt,ii] = sort(-x(jkl));
     jkl = jkl(ii(1:number));
     if  ~sortem,   jkl = sort(jkl);  end    %-- sort by index
  end
  if  sortem
     [tt,ii] = sort(-x(jkl));
     jkl = jkl(ii);
  end
  L = length(jkl);
  peaks(1:L,kk) = x(jkl);
  locs(1:L,kk)  = jkl;
end
