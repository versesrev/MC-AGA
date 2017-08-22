clear;

% Generate low-rank data
dim = [10 10 10 10];
dim2 = [10 5 10 10];
GT = zeros(dim);
GT1 = randn(dim2);
GT2 = randn(dim2);
for k = 3:numel(dim)
    GT1 = fft(GT1,[],k);
    GT2 = fft(GT2,[],k); 
end
for k = 1:prod(dim(3:end))
    GT(:,:,k) = GT1(:,:,k)*GT2(:,:,k)';
end
for k = 3:numel(dim)
    GT = ifft(GT,[],k);
end
GT = real(GT./max(GT(:)));
omega = rand(dim)<0.5;
X = omega.*GT;

% Call MC-AGA
[L,S,iter,obj] = MCAGA(X,1e-7,100);

% Reconstruction error
D = L-GT;
fprintf('Root mean squared error = %f.\n', sqrt(mean(D(:).^2)));
