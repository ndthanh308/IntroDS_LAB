LN_mode = 1;
LN_sigma = 3;
LN_mu = LN_sigma^2 + log(LN_mode);

figure (1);  clf ();

% LEFT: linear scale
alpha = linspace (0, 3, 1e3);
p = lognpdf (alpha, LN_mu, LN_sigma);
subplot (1, 2, 1);  plot (alpha, p, 'LineWidth', 2);
hold on;  plot ([0 0], ylim, 'r--');

% RIGHT: density of log10(alpha)
log10_alpha_mean = LN_mu / (log (10));
log10_alpha_sigma = LN_sigma / (log (10));
log10_alpha = linspace (-1, 9, 1e3);
p = normpdf (log10_alpha, log10_alpha_mean, log10_alpha_sigma);
subplot (1, 2, 2);  plot (log10_alpha, p, 'LineWidth', 2);

csvwrite ('images/PriorPDFlog.csv', [log10_alpha; p]');