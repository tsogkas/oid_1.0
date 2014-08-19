function best = em_hog(X, num_xforms, num_clusters, restarts, max_iters)

best.rho      = [];
best.pi       = [];
best.resp     = [];
best.L        = -inf;
best.Lhistory = [];

for i = 1:restarts
  fprintf('restart %d/%d\n', i, restarts);
  [rho, pi, resp, L] = do_EM(X, num_xforms, num_clusters, max_iters);
  if L > best.L
    best.rho = rho;
    best.pi = pi;
    best.L = L;
    best.resp = resp;
    best.Lhistory = [best.Lhistory L];
    fprintf(' new max obj: %.3f\n', L);
  end
end



function [rho, pi, resp, Lnew] = do_EM(X, num_xforms, num_clusters, max_iters)

LB = 0.1;
UB = 0.9;
LB_DECAY = 1; %0.95;

[dim, num_ex_x_xforms] = size(X);
num = num_ex_x_xforms / num_xforms;

% initialize mixtures using random assignments
count = zeros(1, num_clusters);
rho   = zeros(num_clusters, dim);
p     = zeros(num_clusters, dim);
q     = zeros(num_clusters, dim);
ex_inds = 1:num_xforms:num_ex_x_xforms;
for i = 1:num_xforms:num_ex_x_xforms
  % pick a random component
  k = random('unid', num_clusters);
  %t = random('unid', num_xforms);
  t = 1;
  rho(k,:) = rho(k,:) + X(:,i+t-1)';
  count(k) = count(k) + 1;
end

for j = 1:num_clusters
  rho(j,:) = rho(j,:) / count(j);
  rho(j,:) = truncate(rho(j,:), LB, UB);
  p(j,:) = log(rho(j,:));
  q(j,:) = log(1-rho(j,:));
end
% mixing coefficients
pi = ones(num_clusters, num_xforms);
pi = pi / numel(pi);

% EM
Lold = -inf;
Lnew = Lold;
resp = zeros(num, num_clusters, num_xforms);
for iter = 1:max_iters
  fprintf('iter %d/%d\n', iter, max_iters);
  count = zeros(1, num_clusters);
  rho(:) = 0;

  tic;
  q_sum = sum(q, 2);
  all_logp = (p-q)*X + repmat(q_sum, [1 num_ex_x_xforms]);
  toc;

  tic;
  all_logp = reshape(all_logp, [num_xforms*num_clusters num]);
  logp_max = max(all_logp, [], 1);
  pxi = exp(all_logp - repmat(logp_max, [num_xforms*num_clusters 1]));
  pxi_pi = pxi .* repmat(pi(:), [1 num]);
  inv_Z = 1 ./ sum(pxi_pi, 1);
  gamma = pxi_pi .* repmat(inv_Z, [num_xforms*num_clusters 1]);
  gamma = reshape(gamma, [num_clusters num_xforms*num]);
  count = sum(gamma, 2);
  for k = 1:num_clusters
    rho(k,:) = gamma(k,:)*X' ./ count(k);
    %if t < ceil(max_iters/2)
      rho(k,:) = truncate(rho(k,:), LB, UB);
    %end
  end
  resp = reshape(gamma, [num_clusters num_xforms num]);
  resp = shiftdim(resp, 2); % examples  x  clusters  x  transforms
  toc;
  
%  tic;
%  c = 1;
%  for i = 1:num_xforms:num_ex_x_xforms
%    logp = all_logp(:,i:i+num_xforms-1);
%    % resp are the "responsibilities" or latent variable posteriors
%    resp(c,:,:) = 0;
%    logp_max = max(logp(:));
%    pxi = exp(logp - logp_max);
%    inv_Z = 1 / sum(sum(pxi .* pi));
%    for t = 1:num_xforms
%      xi = X(:,i+t-1);
%      for k = 1:num_clusters
%        % resp(i,t,k) = p(ki = k, ti = t | xi)
%        %             =         p(xi | ki = k, ti = t) * p(ki = k, ti = t)
%        %               ----------------------------------------------------------
%        %               sum_{k',t'} p(xi | ki = k', ti = t') * p(ki = k', ti = t')
%
%        gamma = pxi(k,t) * pi(k,t) * inv_Z;
%        rho(k,:) = rho(k,:) + gamma * xi';
%        count(k) = count(k) + gamma;
%        resp(c,k,t) = gamma;
%      end
%    end
%    c = c + 1;
%  end
%  toc;

  pi = shiftdim(sum(resp, 1), 1)/num;
  pi = max(eps(realmin), pi);
  pi = pi / sum(pi(:));
  for k = 1:num_clusters
    p(k,:) = log(rho(k,:));
    q(k,:) = log(1-rho(k,:));
  end

  try
    nc = 8;
    nr = ceil(num_clusters/nc);
    clf;
    sz = sqrt(size(rho,2)/32);
    for j = 1:num_clusters
      fg = reshape(rho(j,:), [sz sz 32]);
      bg = sum(sum(fg)) / (size(fg, 1) * size(fg, 2));
      bg = repmat(bg, size(fg, 1), size(fg, 2));
      filter = log((fg .* (1-bg))./ ((1-fg) .* bg));
      vl_tightsubplot(nr, nc, j, 'margin', 0.005);
      visualizeHOG(max(0, filter));
    end
    drawnow;
  catch
  end

  % compute obj function value
  % lower bound on log likelihood maximized by EM
  % Lnew = E_q[log L(theta; X,Z)] + H(q)
  tic;
  Lnew = 0;
  q_sum = sum(q,2);
  all_logp = (p-q)*X + repmat(q_sum, [1 num_ex_x_xforms]);
  toc;
  tic;
  c = 1;
  for i = 1:num_xforms:num_ex_x_xforms
    for t = 1:num_xforms
      for k = 1:num_clusters
        Lnew = Lnew + resp(c,k,t) * (all_logp(k,i+t-1) + log(pi(k,t)));
      end
    end

    % bound values away from zero
    r = max(eps(realmin), resp(c,:,:));
    r = r(:)./sum(r(:));
    entropy = -sum(r.*log(r));
    Lnew = Lnew + entropy;
    c = c + 1;
  end
  toc;
  change = (Lnew - Lold)/abs(Lnew);
  fprintf(' obj (lb): %.3f %.4f\n', Lnew, change);
  Lold = Lnew;
  LB = LB * LB_DECAY;

  best.rho  = rho;
  best.pi   = pi;
  best.resp = resp;

  if change < 0.0001
    break;
  end
end


function probs = truncate(probs, lb, ub)
probs = min(ub, max(lb, probs));
