score = importdata('sushi3-2016/sushi3b.5000.10.score',' ');
context = importdata('sushi3-2016/sushi3.idata');

% score: 0,1,2,3,4
% context = [style, major group, minor group,
%            oiliness (0~4), popularity (0~3), price (1~5), availability (0~1)];

context = context.data;
context(:,4) = context(:,4)>2;
context(:,5) = context(:,5)>1.5;
context(:,6) = context(:,6)>2.5;
context(:,7) = context(:,7)>0.5;

[user,item,rating] = find(score+1);
allData = [user,item,context(item,[1,3,4,5,6,7])+1,rating];

Range = max(allData);

N = size(allData,1);
nValid = round(N*0.05);
nTest = round(N*0.2);
nTrain = N-nValid-nTest;

% Load fixed seed for reproducibility
load('rng_state.mat');
rng(state);
idx = randperm(N)';
validIdx = idx(1:nValid);
testIdx = idx(nValid+1:nValid+nTest);
trainIdx = idx(nValid+nTest+1:end);

train = allData(trainIdx,:);
valid = allData(validIdx,:);
test = allData(testIdx,:);

save('sushi_explicit3.mat','trainIdx','validIdx','testIdx','train','valid','test');
