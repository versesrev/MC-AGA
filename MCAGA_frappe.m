% Load frappe dataset
load('CARS2_code/cars2_frappe_datasplit.mat');
load('CARS2_code/cars2_frappe_negids.mat');
load('CARS2_code/cars2_frappe_randitem.mat');

% Convert data to the desired format
train = double(Traindata(:,[1 2 4 3]));
valid = double(Validdata(:,[1 2 4 3]));
test  = double(Testdata(:,[1 2 4 3]));

% Convert a list of subscripts to linear indices
allData = [train;valid;test];
dim = max(allData(:,1:end-1));
indTrain = cellfun(@(x) sub2ind(dim,x{:}),num2cell(num2cell(train(:,1:end-1)),2));
indValid = cellfun(@(x) sub2ind(dim,x{:}),num2cell(num2cell(valid(:,1:end-1)),2));
indTest = cellfun(@(x) sub2ind(dim,x{:}),num2cell(num2cell(test(:,1:end-1)),2));

% Initialize the data matrix
X = zeros(dim);
X(indTrain) = train(:,end);

% Call MC-AGA
[L,S,iter,obj] = MCAGA(X,1e-7,1000,1);

% Evaluate results
results = zeros(2,5);

fpred = L(indValid);
target = (valid(:,end)+1)/2;
results(1,1) = metrics(train,valid,L,dim(2));
[~,results(1,2),results(1,3)] = metrics(train,valid,L,5);
[~,results(1,4),results(1,5)] = metrics(train,valid,L,10);

fpred = L(indTest);
target = (test(:,end)+1)/2;
results(2,1) = metrics(train,test,L,dim(2));
[~,results(2,2),results(2,3)] = metrics(train,test,L,5);
[~,results(2,4),results(2,5)] = metrics(train,test,L,10);

fprintf('iter = %d\n', iter);
fprintf('obj = %.20f\n', obj);
fprintf('map \t mp5 \t mr5 \t mp10 \t mr10\n');
fprintf('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', results(1,:));
fprintf('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', results(2,:));

% Save the learned weights and evaluated results
% Requires a lot of storage space
save('results_frappe.mat','L','S','iter','X','results');
