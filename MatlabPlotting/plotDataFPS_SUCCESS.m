clc;
clear; 
videoName = '06-brzoZaklanjanje';
fileID1 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4boostingMETRIKA5HSVfpssuc.txt'),'r');
fileID2 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4csrtMETRIKA5HSVfpssuc.txt'),'r');
fileID3 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4kcfMETRIKA5HSVfpssuc.txt'),'r');
fileID4 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4medianflowMETRIKA5HSVfpssuc.txt'),'r');
fileID5 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4milMETRIKA5HSVfpssuc.txt'),'r');
fileID6 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4mosseMETRIKA5HSVfpssuc.txt'),'r');
fileID7 = fopen(strcat(strcat('..\BenchmarkResults\Metrika5\Results', videoName),'.mp4tldMETRIKA5HSVfpssuc.txt'),'r');



[nizSucc1, nizFPS1] = getDataForAlgorithm(fileID1);    
[nizSucc2, nizFPS2] = getDataForAlgorithm(fileID2); 
[nizSucc3, nizFPS3] = getDataForAlgorithm(fileID3); 
[nizSucc4, nizFPS4] = getDataForAlgorithm(fileID4); 
[nizSucc5, nizFPS5] = getDataForAlgorithm(fileID5); 
[nizSucc6, nizFPS6] = getDataForAlgorithm(fileID6); 
[nizSucc7, nizFPS7] = getDataForAlgorithm(fileID7); 
plotData(nizSucc1, nizFPS1, ' Boosting');
plotData(nizSucc2, nizFPS2, ' CSRT');
plotData(nizSucc3, nizFPS3, ' KCF');
plotData(nizSucc4, nizFPS4, ' MedianFlow');
plotData(nizSucc5, nizFPS5, ' MIL');
plotData(nizSucc6, nizFPS6, ' MOSSE');
plotData(nizSucc7, nizFPS7, ' TLD');
fclose(fileID1);
fclose(fileID2);
fclose(fileID3);
fclose(fileID4);
fclose(fileID5);
fclose(fileID6);
fclose(fileID7);


function [] = plotData(nizSucc, nizFPS, algorithmName)
    brojRedova = size(nizFPS, 1);
    t = 1:brojRedova;
    subplot(2,1,1)
    plot(t, nizFPS);
    xlabel('Frame');
    ylabel('FPS');
    title(strcat(strcat('Overview of FPS and Success for video 06 and ', algorithmName), ' algorithm'));
    subplot(2, 1, 2)
    plot(t, nizSucc); 
    xlabel('Frame');
    ylabel('Success');
    imeZaFile = extractAfter(algorithmName, 1);
    print(strcat('06_fpssuccess_', imeZaFile), '-depsc');
end

function [nizSucc, nizFPS] = getDataForAlgorithm(fileID)
    nizSucc = [];
    nizFPS = [];
    tline = fgetl(fileID);
    while ischar(tline)
        succ = 0;
        if tline(14) == 'Y'
            succ = 1;
        end
        num = {};
        if ~any(isspace(tline))
           num = regexp(T, '(?<=\.)\d+', 'match');
        end
        numCells = regexp(tline, '\d+', 'match');
        prijeDecimale = str2double(numCells(1));
        poslijeDecimale = str2double(numCells(2));
        cijeliBroj = prijeDecimale*100+poslijeDecimale;
        %size(numCells, 2)
        cijeliBroj = cijeliBroj / 10^size(numCells, 2);
        %succ
        %cijeliBroj
        nizSucc = [nizSucc succ];
        nizFPS = [nizFPS cijeliBroj];
        tline = fgetl(fileID);
    end
    nizFPS = nizFPS';
    nizSucc = nizSucc';
end

