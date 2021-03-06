clear all; 
videoName = '06-brzoZaklanjanje';
fileID1 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4boostingMETRIKA5HSV.txt'),'r');
fileID2 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4csrtMETRIKA5HSV.txt'),'r');
fileID3 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4kcfMETRIKA5HSV.txt'),'r');
fileID4 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4medianflowMETRIKA5HSV.txt'),'r');
fileID5 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4milMETRIKA5HSV.txt'),'r');
fileID6 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4mosseMETRIKA5HSV.txt'),'r');
fileID7 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4tldMETRIKA5HSV.txt'),'r');

formatSpec = '%f';
boosting = fscanf(fileID1, formatSpec);
csrt = fscanf(fileID2, formatSpec);
kcf = fscanf(fileID3, formatSpec);
medianflow = fscanf(fileID4, formatSpec);
mil = fscanf(fileID5, formatSpec);
mosse = fscanf(fileID6, formatSpec);
tld = fscanf(fileID7, formatSpec);
brojRedova = size(boosting, 1);
t = 1:brojRedova;
plot(t, boosting, 'DisplayName','boosting');
hold on;
plot(t, csrt, '-', 'DisplayName','csrt');
plot(t, kcf,  '--','DisplayName','kcf');
plot(t, medianflow, ':',  'DisplayName','medianflow');
plot(t, mil, '-.', 'DisplayName','mil');
plot(t, mosse,  'DisplayName','mosse');
plot(t, tld,  'DisplayName','tld');
legend
xlabel('Frame');
ylabel('Center Location Error');
title('Benchmark results for each algorithm for video 06 with HSV comparison - Center Location Error');
hold off;
print('06_total_centerlocationerror', '-depsc');
fclose(fileID1);
fclose(fileID2);
fclose(fileID3);
fclose(fileID4);
fclose(fileID5);
fclose(fileID6);
fclose(fileID7);