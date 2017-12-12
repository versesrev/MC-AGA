% Load sushi dataset
load('sushi_explicit3.mat');

% Convert a list of subscripts to linear indices
allData = [train;valid;test];
dim = max(allData(:,1:end-1));
indTrain = cellfun(@(x) sub2ind(dim,x{:}),num2cell(num2cell(train(:,1:end-1)),2));
indValid = cellfun(@(x) sub2ind(dim,x{:}),num2cell(num2cell(valid(:,1:end-1)),2));
indTest = cellfun(@(x) sub2ind(dim,x{:}),num2cell(num2cell(test(:,1:end-1)),2));

% Initialize the data matrix
X = zeros(dim);
meanTrain = mean(train(:,end));
X(indTrain) = train(:,end)-meanTrain;

% Call MC-AGA
[L,S,iter,obj] = MCAGA(X,1e-7,1000,1e-4);
L = L+meanTrain;

% Evaluate results
results = zeros(2,2);

fpred = L(indValid);
target = valid(:,end);
results(1,1) = mean(abs(fpred-target));
results(1,2) = sqrt(mean((fpred-target).^2));

fpred = L(indTest);
target = test(:,end);
results(2,1) = mean(abs(fpred-target));
results(2,2) = sqrt(mean((fpred-target).^2));

fprintf('iter = %d\n',iter);
fprintf('obj = %.20f\n',obj);
fprintf('mae \t rmse\n');
fprintf('%.4f\t%.4f\n',results(1,:));
fprintf('%.4f\t%.4f\n',results(2,:));

% Save the learned weights and evaluated results
% Requires a lot of storage space
save('results_sushi.mat','L','S','iter','X','results');
