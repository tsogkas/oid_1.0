function best = EM_Bernoulli(X, num_clusters, restarts, max_iters)

best.rho      = [];
best.pi       = [];
best.resp     = [];
best.L        = -inf;
best.Lhistory = [];

for i = 1:restarts
  fprintf('restart %d/%d\n', i, restarts);
  [rho, pi, resp, L] = do_EM(X, num_clusters, max_iters);
  if L > best.L
    best.rho = rho;
    best.pi = pi;
    best.L = L;
    best.resp = resp;
    best.Lhistory = [best.Lhistory L];
    fprintf(' new max obj: %.3f\n', L);
  end
end



function [rho, pi, resp, Lnew] = do_EM(X, num_clusters, max_iters)

[num, dim] = size(X);

unifp = 1/2^dim;
num_clusters = num_clusters + 1;

% initialize mixtures using random assignments
count = zeros(1, num_clusters);
for j = 1:num_clusters-1
  rho{j} = zeros(1, dim);
end
for i = 1:num
  % pick a random component
  j = random('unid', num_clusters-1);
  rho{j} = rho{j} + X(i,:);
  count(j) = count(j) + 1;
end

for j = 1:num_clusters-1
  rho{j} = rho{j} / count(j);
  rho{j} = truncate(rho{j}, 0.1, 0.9);

  p{j} = log(rho{j});
  q{j} = log(1-rho{j});
end
% mixing coefficients
pi(1:num_clusters-1) = 0.99*ones(1, num_clusters-1)/(num_clusters-1);
pi(num_clusters) = 1 - sum(pi(1:num_clusters-1));


% EM
Lold = -inf;
Lnew = Lold;
resp = zeros(num, num_clusters);
for t = 1:max_iters
  fprintf('iter %d/%d\n', t, max_iters);
  count = zeros(1, num_clusters);
  for j = 1:num_clusters
    rho{j} = zeros(1, dim);
  end
  for i = 1:num
    xi = X(i,:);
    % logp(k) = log p(xi | zi = k) 
    %         = sum_j xi[j]*log(rho[k,j]) + (1-xi[j])*log(1-rho[k,j])
    logp = zeros(1, num_clusters);
    for k = 1:num_clusters-1
      logp(k) = sum(xi .* p{k}) + sum((1-xi) .* q{k});
    end
    logp(num_clusters) = log(unifp);
    % resp are the "responsibilities" or latent variable posteriors
    resp(i,:) = 0;
    logp_max = max(logp(:));
    pxi = exp(logp - logp_max);
    inv_Z = 1 / sum(pxi .* pi);
    for k = 1:num_clusters
      % resp(i,k) = p(zi = k | xi) =
      %           = p(xi | zi = k)*p(zi = k) / sum_q p(xi | zi = q)*p(zi = q)
      gamma = pi(k) * pxi(k) * inv_Z;
      rho{k} = rho{k} + gamma * xi;
      count(k) = count(k) + gamma;
      resp(i,k) = gamma;
    end
  end
  pi = sum(resp)/num;
  pi = max(eps(realmin), pi);
  pi = pi / sum(pi);
  for k = 1:num_clusters-1
    rho{k} = rho{k} / count(k);  
    if t < ceil(max_iters/2)
      rho{k} = truncate(rho{k}, 0.1, 0.9);
    end
    p{k} = log(rho{k});
    q{k} = log(1-rho{k});
  end

  % compute obj function value
  % lower bound on log likelihood maximized by EM
  % Lnew = E_q[log L(theta; X,Z)] + H(q)
  Lnew = 0;
  for i = 1:num
    xi = X(i,:);
    for k = 1:num_clusters-1
      logp = sum(xi .* p{k}) + sum((1-xi) .* q{k});
      Lnew = Lnew + resp(i,k) * (logp + log(pi(k)));
    end
    Lnew = Lnew + resp(i,num_clusters) * (log(unifp) + log(pi(num_clusters)));
    % bound values away from zero
    r = max(eps(realmin), resp(i,:));
    r = r./sum(r);
    entropy = -sum(r.*log(r));
    Lnew = Lnew + entropy;
  end
  change = (Lnew - Lold)/abs(Lnew);
  fprintf(' obj (lb): %.3f %.4f\n', Lnew, change);
  Lold = Lnew;
  if change < 0.0001
    break;
  end
end


function probs = truncate(probs, lb, ub)
probs = min(ub, max(lb, probs));
