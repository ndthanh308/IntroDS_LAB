import torch
from matplotlib import pyplot

def check_nbits(wr, nbits):
    (wr_vals, wr_counts) = torch.unique(wr, sorted=True, return_counts=True)
    assert (len(wr_vals) <= 2**nbits)
    return wr_counts

def round_ldl_gptqequiv(
    w,
    H,
    nbits,
    unbiased=False
):
    """
    w in R^{m,d}
    d: input_shape, m: output_shape
    """
    (d, d_) = H.shape
    assert (d == d_)
    (m, d) = w.shape
    H = torch.flip(H, [0,1])
    L = torch.linalg.cholesky(H)
    L = torch.flip(L,[0,1])
    L = L @ torch.diag(1/torch.diag(L))
    L = L - torch.eye(d)
    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)
    w_hat = w.clone()
    for i in range(d):
        w_hat[:,i] = torch.clamp(torch.floor(w[:,i] + (w - w_hat) @ L[:,i] + eta[:,i]), min=0, max=2**nbits - 1)
    
    wr = w_hat
    wr_counts = check_nbits(wr, nbits)
    return wr

def round_ldl_gptqequiv_noclamp(
    w,
    H,
    nbits,
    unbiased=False
):
    """
    w in R^{m,d}
    d: input_shape, m: output_shape
    """
    (d, d_) = H.shape
    assert (d == d_)
    (m, d) = w.shape
    H = torch.flip(H, [0,1])
    L = torch.linalg.cholesky(H)
    L = torch.flip(L,[0,1])
    L = L @ torch.diag(1/torch.diag(L))
    L = L - torch.eye(d)
    if unbiased:
        eta = torch.rand(w.shape).to(w.device)
    else:
        eta = 0.5 * torch.ones(w.shape).to(w.device)
    w_hat = w.clone()
    for i in range(d):
        w_hat[:,i] = (torch.floor(w[:,i] + (w - w_hat) @ L[:,i] + eta[:,i]))
    
    wr = w_hat
    wr_counts = check_nbits(wr, nbits)
    return wr

def gen_counterexample(m,n,c):
	H = torch.ones(n,n) + torch.eye(n)
	H[n-1,n-1] = 1.0
	H[0,1:(n-1)] += 2 * c
	H[1:(n-1),0] += 2 * c
	H[0,n-1] += c
	H[n-1,0] += c
	H[0,0] += 4 * c + n * (c**2)
	H = H / n
	w = 0.499 * torch.ones(m,n) + 0.002 * (torch.arange(n) % 2)
	return (H,w)

ns = 2**torch.arange(6,13)
m = 16
c = 0.01

loss_ldl_nearest = torch.zeros_like(ns)
loss_ldl_nearest_noclamp = torch.zeros_like(ns)
loss_ldl_stoch = torch.zeros_like(ns)
loss_nearest = torch.zeros_like(ns)
loss_stoch = torch.zeros_like(ns)

for ii in range(len(ns)):
	n = ns[ii]
	print(n)

	(H,w) = gen_counterexample(m, ns[ii], c)

	wr = round_ldl_gptqequiv(w,H,4)
	loss_ldl_nearest[ii] = ((wr - w) @ H @ (wr - w).T).trace()

	wr = round_ldl_gptqequiv_noclamp(w,H,4)
	loss_ldl_nearest_noclamp[ii] = ((wr - w) @ H @ (wr - w).T).trace()

	w_ldl_stoch = round_ldl_gptqequiv(w,H,2,unbiased=True)
	loss_ldl_stoch[ii] = ((w_ldl_stoch - w) @ H @ (w_ldl_stoch - w).T).trace()

	w_nearest = w.round()
	loss_nearest[ii] = ((w_nearest - w) @ H @ (w_nearest - w).T).trace()

	w_stoch = (w + (torch.rand(m,n))).floor()
	loss_stoch[ii] = ((w_stoch - w) @ H @ (w_stoch - w).T).trace()


pyplot.figure(figsize=(4,3))
pyplot.loglog(ns,loss_ldl_nearest,label="LDLQ (nearest)",marker="^")
pyplot.loglog(ns,loss_ldl_stoch,label="LDLQ (stoch)",marker="<")
pyplot.loglog(ns,loss_nearest,label="nearest",marker="s")
pyplot.loglog(ns,loss_stoch,label="stoch",marker="o")
pyplot.loglog(ns,loss_ldl_nearest_noclamp,label="LDLQ (nearest, no clamp)", linestyle=":", marker="*")
pyplot.legend()
pyplot.xlabel("matrix size $n$")
pyplot.ylabel("$\\operatorname{tr}(\\hat W - W) H (\\hat W - W)^T$")
pyplot.tight_layout()
pyplot.savefig("optqbad.pdf")


