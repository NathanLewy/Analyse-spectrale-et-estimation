D = dir('signal');
fid = fopen('signal','r');
signal = fread(fid,D.bytes/4,'float32');
fclose(fid);
% close all
% figure(1), plot(signal)