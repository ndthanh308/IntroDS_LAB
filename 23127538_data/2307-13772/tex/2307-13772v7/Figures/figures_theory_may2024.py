from typing import Any
import warnings
import scipy.optimize as sco
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore")

path = "figures_theory/"
plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)


def settings_plot(ax):
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


class model:

    def __init__(self, h, l, eta, lmb, r, Delta, G):
        self.r = r
        self.h = h
        self.l = l
        self.eta = eta
        self.Delta = Delta
        self.lmb = lmb
        self.G = G

    # function to capture the adverse selection cost on pool with fee f
    def advsel(self, f):
        r, Delta = self.r, self.Delta

        term1 = Delta - (1 + r) * np.sqrt(1 + f)
        term2 = (
            Delta**2 + Delta * (1 + r) * np.sqrt(1 + f) + (1 + f) * (1 + r) * (r - 2)
        )
        term3 = (1 + f) ** (3 / 2) * (r**2) * (r + 1)

        return (term1 * term2 + term3) / (3 * Delta)

    # function to capture the liquidity provision revenue on pool with fee f
    def liqrev(self, f):
        r, Delta = self.r, self.Delta
        numerator = f * (1 + r) * (2 * Delta - r * np.sqrt(1 + f) - 2 * np.sqrt(1 + f))

        return numerator / (Delta)

    # function to capture rebalancing cost
    def rebalancing_cost(self, f):
        r, Delta, G = self.r, self.Delta, self.G
        return G * (1 - (1 + r) * np.sqrt(1 + f) / Delta)

    # PC threshold for pool H
    def qh(self):
        r, h, l, eta, Delta, G = self.r, self.h, self.l, self.eta, self.Delta, self.G

        return (
            eta
            * self.rebalancing_cost(h)
            / ((1 - eta) * self.liqrev(h) - eta * self.advsel(h))
        )

    # PC threshold for pool L
    def ql(self):
        r, h, l, eta, Delta, G = self.r, self.h, self.l, self.eta, self.Delta, self.G

        return (
            eta
            * self.rebalancing_cost(l)
            / ((1 - eta) * self.liqrev(l) - eta * self.advsel(l))
        )

    # marginal trader in fragmented equilibrium
    def qmg_eq(self):
        r, h, l, eta, Delta, G = self.r, self.h, self.l, self.eta, self.Delta, self.G

        numerator = eta * (self.rebalancing_cost(h) - self.rebalancing_cost(l))
        denominator = ((1 - eta) * self.liqrev(l) - eta * self.advsel(l)) - (
            (1 - eta) * self.liqrev(h) - eta * self.advsel(h)
        )

        return numerator / denominator

    def liquidity_amounts(self):
        lmb = self.lmb

        qmg = self.qmg_eq()
        qh = self.qh()
        ql = self.ql()

        if qmg < 0:
            return np.array([0, (lmb + qh) * np.exp(-1 / lmb * qh)])
        elif qh > ql:
            return np.array([(lmb + ql) * np.exp(-1 / lmb * ql), 0])
        else:
            return np.array(
                [
                    (lmb + qmg) * np.exp(-1 / lmb * (qmg)),
                    (lmb + qh) * np.exp(-1 / lmb * qh)
                    - (lmb + qmg) * np.exp(-1 / lmb * qmg),
                ]
            )

    # market shares in fragmented equilibrium
    def market_shares(self):
        liq = self.liquidity_amounts()
        return liq / np.sum(liq)


r, h, l, lmb, G, eta = (
    1.001,
    3,
    1,
    1,
    50,
    0.3,
)
Delta = 1.1 * (1 + r) * np.sqrt(1 + h)


sizeOfFont = 18
ticks_font = font_manager.FontProperties(size=sizeOfFont)
sizefigs_L = (14, 6)
gs = gridspec.GridSpec(1, 2)
fig = plt.figure(facecolor="white", figsize=sizefigs_L)

ax = fig.add_subplot(gs[0, 0])
ax = settings_plot(ax)

G_space = np.linspace(0.01, 50, 1000)
wL_space = [model(h, l, eta, lmb, r, Delta, gi).market_shares()[0] for gi in G_space]
wH_space = [model(h, l, eta, lmb, r, Delta, gi).market_shares()[1] for gi in G_space]

plt.plot(G_space, wL_space, label=r"Liquidity pool $L$", c="r")
plt.plot(G_space, wH_space, label=r"Liquidity pool $H$", c="b", ls="--")

plt.ylabel("Market share ($w_k$)", fontsize=18)
plt.xlabel("Gas cost ($\Gamma$)", fontsize=18)
plt.legend(loc="best", fontsize=18, frameon=False)


ax = fig.add_subplot(gs[0, 1])
ax = settings_plot(ax)

G_space = np.linspace(0.01, 50, 1000)
LL_space = [
    model(h, l, eta, lmb, r, Delta, gi).liquidity_amounts()[0] for gi in G_space
]
LH_space = [
    model(h, l, eta, lmb, r, Delta, gi).liquidity_amounts()[1] for gi in G_space
]

# plt.plot(G_space, LL_space, label=r"Liquidity pool $L$", c="r")
plt.plot(
    G_space,
    np.array(LL_space) + np.array(LH_space),
    label=r"Total liquidity supply",
    c="b",
    ls="-",
)
# plt.axvline(x=h, c="k", lw=0.5, ls="-.")
# plt.text(x=h + 0.02, y=0.8, s=r"$\Gamma=h$", fontsize=18)

# plt.axvline(x=h - (h - l) * Q ** ((lmb / th) * (Q / (Q - 1))), c="k", lw=0.5, ls="-.")
# plt.text(
#     x=h - (h - l) * Q ** ((lmb / th) * (Q / (Q - 1))) + 0.02,
#     y=1.1,
#     s=r"$\Gamma=h-\left(h-\ell\right) Q^{\frac{\lambda}{\theta}\frac{Q}{Q-1}}$",
#     fontsize=18,
# )

# plt.axvline(x=Q * l, c="k", lw=0.5, ls="-.")
# plt.text(x=Q * l + 0.02, y=0.8, s=r"$\Gamma=Q\ell$", fontsize=18)
plt.ylabel("Liquidity deposit ($\sum_k T_k$)", fontsize=18)
plt.xlabel("Gas cost ($\Gamma$)", fontsize=18)
plt.legend(loc="best", fontsize=18, frameon=False)

plt.tight_layout(pad=3.0)
# plt.show()
plt.savefig(path + "mkt_share_gas_may2024.png", bbox_inches="tight")

ggg

## Conditions for equilibrium existence
## ---------------------------------------------

th = 2 / 3  # update this parameter


def cond_Q(Q, h, l, lmb, th, G):
    x = sco.fsolve(
        lambda x: 1 - (h - l) / h * x ** ((lmb / th) * (x / (x - 1))), (1 + Q) / 2
    )
    return x


def cond_G(Q, h, l, lmb, th, G):
    return np.maximum(0, h - (h - l) * Q ** ((lmb / th) * (Q / (Q - 1))))


Qspace = np.linspace(1.1, 6, 1000)


sizeOfFont = 18
ticks_font = font_manager.FontProperties(size=sizeOfFont)
sizefigs_L = (16, 9)
gs = gridspec.GridSpec(1, 1)
fig = plt.figure(facecolor="white", figsize=sizefigs_L)

ax = fig.add_subplot(gs[0, 0])
ax = settings_plot(ax)

plt.plot(Qspace, [cond_G(qi, h, l, lmb, th, G) for qi in Qspace], c="r")
plt.vlines(x=cond_Q(Q, h, l, lmb, th, G), ymin=0, ymax=5, colors="k", ls="-.", lw=0.75)
plt.axhline(y=0, lw=0.5, ls="-.", c="k")
plt.plot(Qspace, [qi * l for qi in Qspace], c="b")

ax.fill_between(
    Qspace,
    [cond_G(qi, h, l, lmb, th, G) for qi in Qspace],
    [qi * l for qi in Qspace],
    color="b",
    alpha=0.2,
)

plt.text(x=1.13, y=0.12, s=r"Pool $L$ only in equilibrium ", fontsize=18)
plt.text(x=1.25, y=4, s=r"Pool $H$ only in equilibrium", fontsize=18)
plt.text(
    x=1.8,
    y=0.8,
    s=r"Fragmentation: both pools have positive liquidity in equilibrium",
    fontsize=18,
)

plt.text(
    x=cond_Q(Q, h, l, lmb, th, G) + 0.02,
    y=4,
    s=r"$\frac{h-\ell}{h} Q^{\frac{\lambda}{\theta}\frac{Q}{Q-1}}-1=0$",
    c="k",
    fontsize=18,
)

plt.text(
    x=4.2,
    y=0.1,
    s=r"$\Gamma=h-\left(h-\ell\right) Q^{\frac{\lambda}{\theta}\frac{Q}{Q-1}}$",
    c="r",
    fontsize=18,
)
plt.text(x=2, y=1.75, s=r"$\Gamma=Q\ell$", c="b", fontsize=18)


plt.xlabel(r"Liquidity provider maximum endowment ($Q$)", fontsize=18)
plt.ylabel("Gas cost ($\Gamma$)", fontsize=18)

plt.xlim(1.1, 6)
plt.tight_layout(pad=3.0)
# plt.show()
plt.savefig(path + "equilibrium_existence.png", bbox_inches="tight")


def phi(lmbda, Q):
    temp = Q / (Q - 1)

    if lmbda < 1:
        return 0
    elif lmbda > Q:
        return 0
    else:
        return temp / (lmbda**2)


sizeOfFont = 18
ticks_font = font_manager.FontProperties(size=sizeOfFont)

sizefigs_L = (12, 7)

Q_vars = [3, 4]

lmbda_space = np.linspace(1 - 0.1, Q_vars[1] + 0.1, 1000)

gs = gridspec.GridSpec(1, 1)

fig = plt.figure(facecolor="white", figsize=sizefigs_L)

ax = fig.add_subplot(gs[0, 0])
ax = settings_plot(ax)

plt.plot(
    lmbda_space,
    [phi(lmbdai, Q_vars[0]) for lmbdai in lmbda_space],
    c="b",
    ls="-",
    lw=2,
    label=r"Q=%2.f" % Q_vars[0],
)
plt.plot(
    lmbda_space,
    [phi(lmbdai, Q_vars[1]) for lmbdai in lmbda_space],
    c="r",
    ls="--",
    lw=2,
    label=r"Q=%2.f" % Q_vars[1],
)

plt.legend(loc="best", fontsize=18, frameon=False)

plt.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    top=False,
)  # ticks along the top edge are off

plt.tick_params(
    axis="y",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    right=False,
)  # ticks along the top edge are off

plt.xlabel(r"Liquidity provider token endowment ($q_i$)", fontsize=20)
plt.ylabel(r"Density: $\varphi\left(q_i\right)$", fontsize=20)

# plt.show()
plt.savefig(path + "model_density.png", bbox_inches="tight")

## Aggregate volume simulation
##------


sim = 100000
dt = 0.1
arrivals = 1 * (np.random.rand(sim) < 1 - np.exp(-lmb * dt))

vol_L = th * dt + model(Q, h, l, lmb, th, G).liquidity_amounts()[0] * arrivals
vol_H = model(Q, h, l, lmb, th, G).liquidity_amounts()[1] * arrivals

df = pd.DataFrame(np.array([vol_L, vol_H]).T)
df = df.rename(columns={0: r"Pool $L$", 1: r"Pool $H$"})
df2 = df.stack().reset_index()
df2 = df2.rename(columns={"level_1": "pool", 0: "trade"})
df2["tradeunit"] = df2["trade"] / dt
df2["trade"] = np.where(df2["trade"] == 0, np.nan, df2["trade"])

gs = gridspec.GridSpec(1, 2)

fig = plt.figure(facecolor="white", figsize=(18, 9))

ax = fig.add_subplot(gs[0, 0])
ax = settings_plot(ax)

sns.barplot(data=df2, x="pool", y="tradeunit", palette="Blues")
plt.xlabel("Liquidity pool", fontsize=18)
plt.ylabel("Trade volume per unit of time", fontsize=18)
plt.title("Trade volume on fragmented pools", fontsize=18)

ax = fig.add_subplot(gs[0, 1])
ax = settings_plot(ax)

sns.barplot(data=df2, x="pool", y="trade", palette="Blues")

plt.xlabel("Liquidity pool", fontsize=18)
plt.ylabel("Trade size", fontsize=18)
plt.title("Average trade size on fragmented pools", fontsize=18)

plt.tight_layout(pad=3.0)
# plt.show()
plt.savefig(path + "trade_volumesize.png", bbox_inches="tight")

# Compute conditional expectations of liquidity mint


def inverse_varphi(u, Q):
    return 1 / (1 - ((Q - 1) / Q) * u)


G = 1.2

liq_amounts = inverse_varphi(np.random.rand(sim), Q)
qmg = model(Q, h, l, lmb, th, G).qmg_eq()
df_liq = pd.DataFrame(liq_amounts, columns={"qi"})
df_liq["Liquidity pool"] = np.where(
    df_liq["qi"] <= G / h,
    "Not providing liquidity",
    np.where(df_liq["qi"] <= qmg, "Pool H", "Pool L"),
)

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(facecolor="white", figsize=(16, 9))

ax = fig.add_subplot(gs[0, :])
ax = settings_plot(ax)

sns.histplot(data=df_liq, x="qi", hue="Liquidity pool", bins=200, stat="percent")

legend = ax.get_legend()
handles = legend.legendHandles
legend.remove()
ax.legend(
    handles,
    ["Pool H", "Pool L", "Not providing liquidity"],
    title="Liquidity pool",
    fontsize=16,
    frameon=False,
    title_fontsize=18,
)

plt.xlabel(r"Liquidity endowment ($q_i$)", fontsize=18)
plt.ylabel(r"Percent", fontsize=18)

df_liq = df_liq[df_liq["Liquidity pool"].isin(["Pool H", "Pool L"])]

ax = fig.add_subplot(gs[1, 0])
ax = settings_plot(ax)

sns.barplot(data=df_liq, x="Liquidity pool", y="qi", palette="Blues")
plt.ylabel(r"Liquidity endowment ($q_i$)", fontsize=18)
plt.xlabel(r"Liquidity pool", fontsize=18)

ax = fig.add_subplot(gs[1, 1])
ax = settings_plot(ax)

sns.countplot(data=df_liq, x="Liquidity pool", palette="Blues")
plt.ylabel(r"Number of \textbf{LP}s (per 100,000)", fontsize=18)
plt.xlabel(r"Liquidity pool", fontsize=18)

plt.tight_layout(pad=3.0)
# plt.show()
plt.savefig(path + "liquidity_supply.png", bbox_inches="tight")


#### Trials w/ Q

# Q, h, l, lmb, th, G = 2, 1.5, 1.2, 0.4, 0.5, 1.5
Q, h, l, lmb, th, G = 3, 1.5, 0.75, 0.5, 1, 0.5


sizeOfFont = 18
ticks_font = font_manager.FontProperties(size=sizeOfFont)
sizefigs_L = (18, 9)
gs = gridspec.GridSpec(1, 2)
fig = plt.figure(facecolor="white", figsize=sizefigs_L)

ax = fig.add_subplot(gs[0, 0])
ax = settings_plot(ax)

Q_space = np.linspace(1.5, 5, 1000)
wL_space = [model(Qi, h, l, lmb, th, G).market_shares()[0] for Qi in Q_space]
wH_space = [model(Qi, h, l, lmb, th, G).market_shares()[1] for Qi in Q_space]

plt.plot(Q_space, wL_space, label=r"Liquidity pool $L$", c="r")
plt.plot(Q_space, wH_space, label=r"Liquidity pool $H$", c="b", ls="--")
# plt.axvline(x=h,c='k',lw=0.5,ls='-.')
# plt.text(x=h+0.02,y=0.8,s=r"$\Gamma=h$",fontsize=18)
# plt.axvline(x=Q*l,c='k',lw=0.5,ls='-.')
# plt.text(x=Q*l+0.02,y=0.8,s=r"$\Gamma=Q\ell$",fontsize=18)

# plt.axvline(x=h-(h-l)*Q**((lmb/th)*(Q/(Q-1))),c='k',lw=0.5,ls='-.')
# plt.text(x=h-(h-l)*Q**((lmb/th)*(Q/(Q-1)))+0.02,y=1.02,s=r"$\Gamma=h-\left(h-\ell\right) Q^{\frac{\lambda}{\theta}\frac{Q}{Q-1}}$",fontsize=18)


plt.ylabel("Market share ($w_i$)", fontsize=18)
plt.xlabel("Liquidity supply heterogeneity ($Q$)", fontsize=18)
plt.legend(loc="best", fontsize=18, frameon=False)


ax = fig.add_subplot(gs[0, 1])
ax = settings_plot(ax)

LL_space = [model(Qi, h, l, lmb, th, G).liquidity_amounts()[0] for Qi in Q_space]
LH_space = [model(Qi, h, l, lmb, th, G).liquidity_amounts()[1] for Qi in Q_space]

plt.plot(Q_space, LL_space, label=r"Liquidity pool $L$", c="r")
plt.plot(Q_space, LH_space, label=r"Liquidity pool $H$", c="b", ls="--")
# plt.axvline(x=h,c='k',lw=0.5,ls='-.')
# plt.text(x=h+0.02,y=0.8,s=r"$\Gamma=h$",fontsize=18)

# plt.axvline(x=h-(h-l)*Q**((lmb/th)*(Q/(Q-1))),c='k',lw=0.5,ls='-.')
# plt.text(x=h-(h-l)*Q**((lmb/th)*(Q/(Q-1)))+0.02,y=1.1,s=r"$\Gamma=h-\left(h-\ell\right) Q^{\frac{\lambda}{\theta}\frac{Q}{Q-1}}$",fontsize=18)

# plt.axvline(x=Q*l,c='k',lw=0.5,ls='-.')
# plt.text(x=Q*l+0.02,y=0.8,s=r"$\Gamma=Q\ell$",fontsize=18)
plt.ylabel("Liquidity deposit ($\mathcal{L}_i$)", fontsize=18)
plt.xlabel("Liquidity supply heterogeneity ($Q$)", fontsize=18)
plt.legend(loc="best", fontsize=18, frameon=False)

plt.tight_layout(pad=3.0)
# plt.show()
plt.savefig(path + "mkt_share_Q.png", bbox_inches="tight")
