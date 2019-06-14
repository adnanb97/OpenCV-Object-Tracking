clc;
clear; 
imeFajla = input('Unesite ime fajla: ', 's');
fileID = fopen(strcat('..\BenchmarkResults\', imeFajla),'r');

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

brojRedova = size(nizFPS, 1);
t = 1:brojRedova;
subplot(2,1,1)
plot(t, nizFPS);
xlabel('Frame');
ylabel('FPS');
title('Overview of FPS and Success for TLD algorithm');
subplot(2, 1, 2)
plot(t, nizSucc); 
xlabel('Frame');
ylabel('Success');
fclose(fileID);

