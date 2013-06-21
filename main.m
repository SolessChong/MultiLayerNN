
  
clc; 
close all;
clear all;
  
input = [-2 : 0.4 : 2;-2:0.4:2]; 
ican = 4;
  
depth = 4;  % total layer - 1, by convension
[featureNum , sampleNum] = size(input); 
levelNum(1) = featureNum;  
levelNum(2) = 5;
levelNum(3) = 5;
levelNum(4) = 5;
levelNum(5) = 2;


weight = cell(0);
for k = 1 : depth
    weight{k} = rand(levelNum(k+1), levelNum(k)) - 2 * rand(levelNum(k+1) , levelNum(k));
    threshold{k} = rand(levelNum(k+1) , 1) - 2 * rand(levelNum(k+1) , 1);
end
  
runCount = 0;
sumMSE = 1; % init MSE 
minError = 1e-5;

afa = 0.1; % step of "gradient ascendence"
  
% training loop
while(runCount < 100000 & sumMSE > minError)  
    sumMSE = 0; % sum of MSE
    for i = 1 : sampleNum % sample loop
        netValue{1} = input(:,i);
        
        for k = 2 : depth
            netValue{k} = weight{k-1} * netValue{k-1} + threshold{k-1}; %calculate each layer 
            netValue{k} = 1 ./ (1 + exp(-netValue{k})); %apply logistic function 
        end
        netValue{depth+1} = weight{depth} * netValue{depth} + threshold{depth}; %output layer
        
        
        e = 1 + sin((pi / 4) * ican * netValue{1}) - netValue{depth + 1}; %calc error
        assistS{depth} = diag(ones(size(netValue{depth+1})));
        s{depth} = -2 * assistS{depth} * e;
        for k = depth - 1 : -1 : 1
            assistS{k} = diag((1-netValue{k+1}).*netValue{k+1});
            s{k} = assistS{k} * weight{k+1}' * s{k+1};  
        end
          
        for k = 1 : depth
            weight{k} = weight{k} - afa * s{k} * netValue{k}';
            threshold{k} = threshold{k} - afa * s{k};
        end

        sumMSE = sumMSE + e' * e;  
    end  
    sumMSE = sqrt(sumMSE) / sampleNum;  
    runCount = runCount + 1;  
end  
  
x = [-2 : 0.1 : 2;-2:0.1:2];  
y = zeros(size(x));  
z = 1 + sin((pi / 4) * ican .* x);  
% test
for i = 1 : length(x)  
    netValue{1} = x(:,i);
    for k = 2 : depth
        netValue{k} = weight{k-1} * netValue{k-1} + threshold{k-1};
        netValue{k} = 1 ./ ( 1 + exp(-netValue{k}));
    end
    y(:, i) = weight{depth} * netValue{depth} + threshold{depth};
end  

plot(x(1,:) , y(1,:) , 'r');  
hold on;  

plot(x(1,:) , z(1,:) , 'g');  

hold off;  