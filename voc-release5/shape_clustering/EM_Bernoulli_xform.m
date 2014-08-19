function best = EM_Bernoulli_xform(X, num_clusters, restarts, max_iters, save_fn)

best.rho      = [];
best.pi       = [];
best.resp     = [];
best.L        = -inf;
best.Lhistory = [];

for i = 1:restarts
  fprintf('restart %d/%d\n', i, restarts);
  [rho, pi, resp, L] = do_EM(X, num_clusters, max_iters, save_fn);
  if L > best.L
    best.rho = rho;
    best.pi = pi;
    best.L = L;
    best.resp = resp;
    best.Lhistory = [best.Lhistory L];
    fprintf(' new max obj: %.3f\n', L);
    save_fn(best);
  end
end



function [rho, pi, resp, Lnew] = do_EM(X, num_clusters, max_iters, save_fn)

LB = 0.002;
UB = 0.95;
LB_DECAY = 1; %0.95;

[num, num_xforms] = size(X);

dim = 100*100;

mu = max(cellfun(@length, X));
num_clusters = num_clusters;

% initialize mixtures using random assignments
count = zeros(1, num_clusters);
for j = 1:num_clusters
  rho{j} = zeros(dim, 1);
end
for i = 1:num
  % pick a random component
  j = random('unid', num_clusters);
  xtI = X{i,1};
  rho{j}(xtI) = rho{j}(xtI) + 1;
  count(j) = count(j) + 1;
end

for j = 1:num_clusters
  rho{j} = rho{j} / count(j);
  rho{j} = truncate(rho{j}, LB, UB);

  p{j} = log(rho{j});
  q{j} = log(1-rho{j});
end
% mixing coefficients
pi = ones(num_xforms, num_clusters);
pi = pi / numel(pi);

% EM
Lold = -inf;
Lnew = Lold;
resp = zeros(num, num_xforms, num_clusters);
for iter = 1:max_iters
  fprintf('iter %d/%d\n', iter, max_iters);
  count = zeros(1, num_clusters);
  for j = 1:num_clusters
    rho{j} = zeros(dim, 1);
    q_sum{j} = sum(q{j}(:));
  end
  for i = 1:num
    % logp(k,t) = log p(xi | ki = k, ti = t)
    %           = sum_j xit[i]*log(rho[k,j]) + (1-xit[j]*log(1-rho[k,j])
    logp = zeros(num_xforms, num_clusters);
    for t = 1:num_xforms
      xtI = X{i,t};
      %xtnotI = setdiff(1:dim, xtI);
      for k = 1:num_clusters
        %logp(t,k) = sum(p{k}(xtI)) + sum(q{k}(xtnotI));
        logp(t,k) = sum(p{k}(xtI)) + q_sum{k} - sum(q{k}(xtI));
      end
    end

    % resp are the "responsibilities" or latent variable posteriors
    resp(i,:,:) = 0;
    logp_max = max(logp(:));
    pxi = exp(logp - logp_max);
    inv_Z = 1 / sum(sum(pxi .* pi));
    for t = 1:num_xforms
      xtI = X{i,t};
      for k = 1:num_clusters
        % resp(i,t,k) = p(ki = k, ti = t | xi)
        %             =         p(xi | ki = k, ti = t) * p(ki = k, ti = t)
        %               ----------------------------------------------------------
        %               sum_{k',t'} p(xi | ki = k', ti = t') * p(ki = k', ti = t')

        gamma = pxi(t,k) * pi(t,k) * inv_Z;
        %rho{k} = rho{k} + gamma * xi;
        rho{k}(xtI) = rho{k}(xtI) + gamma;
        count(k) = count(k) + gamma;
        resp(i,t,k) = gamma;
      end
    end
  end
  %keyboard
  %pi = squeeze(sum(resp))/num;
  pi = sum(shiftdim(resp,1),3)/num;
  pi = max(eps(realmin), pi);
  pi = pi / sum(sum(pi));
  for k = 1:num_clusters
    rho{k} = rho{k} / count(k);  
    %if t < ceil(max_iters/2)
      rho{k} = truncate(rho{k}, LB, UB);
    %end
    p{k} = log(rho{k});
    q{k} = log(1-rho{k});
    q_sum{k} = sum(q{k}(:));
  end

  figure(iter);
  R = cellfun(@(x) reshape(x, [100 100 1]), rho, 'UniformOutput', false);
  montage(cat(4, R{:}));
  colormap hot;
  drawnow;

  % compute obj function value
  % lower bound on log likelihood maximized by EM
  % Lnew = E_q[log L(theta; X,Z)] + H(q)
  Lnew = 0;
  for i = 1:num
    for t = 1:num_xforms
      xtI = X{i,t};
      %xtnotI = setdiff(1:dim, xtI);
      for k = 1:num_clusters
        %logp = sum(xi .* p{k}) + sum((1-xi) .* q{k});
        %logp = sum(p{k}(xtI)) + sum(q{k}(xtnotI));
        logp = sum(p{k}(xtI)) + q_sum{k} - sum(q{k}(xtI));
        Lnew = Lnew + resp(i,t,k) * (logp + log(pi(t,k)));
      end
    end

    % bound values away from zero
    r = max(eps(realmin), resp(i,:,:));
    r = r(:)./sum(r(:));
    entropy = -sum(r.*log(r));
    Lnew = Lnew + entropy;
  end
  change = (Lnew - Lold)/abs(Lnew);
  fprintf(' obj (lb): %.3f %.4f\n', Lnew, change);
  Lold = Lnew;
  LB = LB * LB_DECAY;

  best.rho = rho;
  best.pi = pi;
  best.resp = resp;
  save_fn(best);

  if change < 0.0001
    break;
  end
end


function probs = truncate(probs, lb, ub)
probs = min(ub, max(lb, probs));
