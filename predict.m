load('HW_train.mat');

%% predict
x = data(1:4,trainIndicator<trainPortion); 
y = zeros(1, size(x,2));
z = data(5,trainIndicator<trainPortion);
%{
%No.1
x = data;
y = zeros(size(x));  
%}
%{
No.2
x = [-2 : 0.1 : 2; -2 : 0.1 : 2];
y = zeros(size(x));
z = sin(x * 2);
%}

% test output
for i = 1 : length(x)  
    netValue{1} = x(:,i);
    for k = 2 : depth
        netValue{k} = weight{k-1} * netValue{k-1} + threshold{k-1};
        netValue{k} = 1 ./ ( 1 + exp(-netValue{k}));
    end
    y(:, i) = weight{depth} * netValue{depth} + threshold{depth};
end  

% predict
class = round(y);
correct = (class==z);
disp(['correct :' int2str(sum(correct)) ' in ' int2str(length(y)) ' test samples']);
disp(['correct ratio: ' num2str(sum(correct) / length(y))]);
%% plot 

plot(y, 'r');
hold on
plot(z, 'b');
hold off


%{
%No.1
scatter(x(1,:), x(2,:), 'r');
hold on;  

scatter(y(1,:), y(2,:), 'b');
hold off;  
%}

%{  
%No.2
plot(x(2,:), y(2,:),'r');
hold on;  

plot(x(2,:), z(2,:), 'bo');
hold off;  
%}