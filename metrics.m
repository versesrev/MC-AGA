function [map,mp,mr] = metrics(trainT,testT,fpred,topn)

% Load negative samples for metrics
load('CARS2_code/cars2_frappe_randitem.mat');

allData = double([trainT;testT]);
dim = max(allData);
M = dim(1);
N = dim(2);

% Enumerate contexts
allContext = allData(:,3:end-1);
[~,~,ctxId] = unique(allContext,'rows');
trainCId = ctxId(1:size(trainT,1));
testCId = ctxId(size(trainT,1)+1:end);

Traindata = double([trainT(:,[1 2]) (trainT(:,end)>0) trainCId]);
Testdata = double([testT(:,[1 2]) (testT(:,end)>0) testCId]);

relval = 1;
aps = [];
precs = [];
recs = [];
for k = 1:max(ctxId)
    ind = find(Testdata(:,4)==k);
    if ~isempty(ind)
        % Prevent testing on training data
        indtr = find(Traindata(:,4)==k);
        if ~isempty(indtr)
            tr = sparse(Traindata(indtr,1),Traindata(indtr,2),Traindata(indtr,3),M,N);
        else
            tr = sparse(M,N);
        end
        pred = fpred(:,:,k);
        pred(tr>0) = 0;

        ts = sparse(Testdata(ind,1),Testdata(ind,2),Testdata(ind,3),M,N);
        for j = 1:M
            % Evaluate every (user, context) pair separately
            ind = find(ts(j,:)>0);
            if length(ind)>0
                % Include randomly sampled items as negative samples
                indtest = union(ind,indrand);

                ap = ap_metric(pred(j,indtest),ts(j,indtest),topn,relval);
                aps = [aps,ap];
                [r,p] = score(pred(j,indtest),ts(j,indtest),topn,relval);
                precs = [precs,p];
                recs = [recs,r];
            end
        end
    end
end
map = mean(aps);
mp = mean(precs);
mr = mean(recs);
end

function [rec,prec] = score(pred,relevant,topn,relval)
[~,predId] = sort(pred,'descend');
relevantId = find(relevant >= relval);
len = min(numel(predId),topn);
rec = numel(intersect(predId(1:len),relevantId))/numel(relevantId);
prec = numel(intersect(predId(1:len),relevantId))/topn;
end

function ap = ap_metric(pred,Testdata,topn,relval)
% ap : average precision for each query
% pred: prediction matrix
% Testdata: Groundtruth matrix
% relval: The threshold for relevance

num_hits = 0;
s = 0;
ind = find(Testdata>=relval);
if isempty(ind)
    ap = 0;
else
    [~,nb] = sort(full(pred),'descend');
    for i = 1:min(length(nb),topn)
        if ismember(nb(i),ind)
            num_hits = num_hits+1;
            s = s+num_hits/i;
        end
    end
    ap = s/min(length(ind),topn);
end
end
