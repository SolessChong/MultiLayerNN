
  
clc; 
close all;
clear all;
  
%% init


file = fopen('classdata.txt');
data = fscanf(file, '%f,%f,%f,%f,%d', [5, inf]);

trainIndicator = rand(1, size(data, 2));
trainPortion = 0.15;
input = data(1:4,trainIndicator > trainPortion);
output = data(1:4,trainIndicator > trainPortion);


%{
input = [-2 : 0.4 : 2;-2:0.4:2]; 
output = sin(input);
output(2,:) = sin(input(2,:) * 2);
%}
  
depth = 4;  % total layer - 1, by convension
[featureNum , sampleNum] = size(input); 
levelNum(1) = featureNum;  
levelNum(2) = 7;
levelNum(3) = 1;
levelNum(4) = 7;
levelNum(5) = featureNum;


weight = cell(depth);
threshold = cell(depth);
for k = 1 : depth
    weight{k} = rand(levelNum(k+1), levelNum(k)) - 2 * rand(levelNum(k+1) , levelNum(k));
    threshold{k} = rand(levelNum(k+1) , 1) - 2 * rand(levelNum(k+1) , 1);
end
  
runCount = 0;
sumMSE = 1; % init MSE 
minError = 1e-5;

afa = 0.1; % step of "gradient ascendence"
mAlpha = 0.2; % ratio of momentum
  
%% training loop
oldDW = cell(depth);
oldDT = cell(depth);
netValue = cell(depth);
assistS = cell(depth);
s = cell(depth);
    % momentum
for i = 1 : depth
    oldDW{i} = 0;
    oldDT{i} = 0;
end
while(runCount < 100000 && sumMSE > minError)  
    sumMSE = 0; % sum of MSE
    for i = 1 : sampleNum % sample loop
        netValue{1} = input(:,i);
        
        % calculat the network
        for k = 2 : depth
            netValue{k} = weight{k-1} * netValue{k-1} + threshold{k-1}; %calculate each layer 
            netValue{k} = 1 ./ (1 + exp(-netValue{k})); %apply logistic function 
        end
        netValue{depth+1} = weight{depth} * netValue{depth} + threshold{depth}; %output layer
        
        % update the weights
        e = output(:,i) - netValue{depth + 1}; %calc error
        assistS{depth} = diag(ones(size(netValue{depth+1})));
        s{depth} = -2 * assistS{depth} * e;
        for k = depth - 1 : -1 : 1
            assistS{k} = diag((1-netValue{k+1}).*netValue{k+1});
            s{k} = assistS{k} * weight{k+1}' * s{k+1};  
        end
          
        
        for k = 1 : depth
            dW = (1-mAlpha) * s{k} * netValue{k}' + mAlpha * oldDW{k};
            dT = (1-mAlpha) * s{k} + mAlpha * oldDT{k};
            oldDW{k} = dW; oldDT{k} = dT;
            
            weight{k} = weight{k} - afa * dW;
            threshold{k} = threshold{k} - afa * dT;
        end

        sumMSE = sumMSE + e' * e;  
        
        % count training iteration
        runCount = runCount + 1;  
    end  
    sumMSE = sqrt(sumMSE) / sampleNum;  
    
end  
  