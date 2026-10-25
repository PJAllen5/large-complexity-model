"""Stochastic simulators of complex systems from several fields.

Every simulator follows one interface so the rest of the code never needs
to know which field a system comes from:

    state, params = sim.init(rng, B)        # B independent episodes
    state = sim.step(state, params, rng)    # advance one small time step
    obs = sim.obs(state, params)            # -> [B, N, F] observed numbers
    params["adj"]                           # -> [B, N, N] who interacts with whom

Each episode draws its own random parameters (food web, coupling strength,
infection rate, ...), so the model sees many variants of every system.
"""

import numpy as np


def _sym_er(rng, B, N, p):
    """Random symmetric interaction graph (no self-links)."""
    p = np.asarray(p).reshape(-1, 1, 1) if np.ndim(p) else p
    up = np.triu(rng.random((B, N, N)) < p, 1)
    return (up | up.transpose(0, 2, 1)).astype(np.float64)


class Simulator:
    name = ""
    field = ""
    n = 0          # number of entities
    f = 0          # features per entity
    substeps = 1   # small steps per observation
    max_burn = 30  # random warm-up observations before recording
    event_f = 0    # feature used for the "will it go up?" likelihood test
    feature_names = ()

    def init(self, rng, B):
        raise NotImplementedError

    def step(self, state, params, rng):
        raise NotImplementedError

    def obs(self, state, params):
        raise NotImplementedError

    def run(self, state, params, rng, T):
        """Advance T observations. Returns final state and obs [B, T, N, F]."""
        out = []
        for _ in range(T):
            for _ in range(self.substeps):
                state = self.step(state, params, rng)
            out.append(self.obs(state, params))
        return state, (np.stack(out, 1) if out else None)


# ---------------------------------------------------------------- ecology
class PredatorPrey(Simulator):
    """Food web of 6 species (generalised Lotka-Volterra with noise)."""
    name, field, n, f, substeps = "predator_prey", "ecology", 6, 1, 4
    feature_names = ("log abundance",)
    dt = 0.05

    def init(self, rng, B):
        N = self.n
        idx = np.arange(N)
        n_basal = rng.integers(2, 5, size=(B, 1))
        basal = idx[None] < n_basal
        r = np.where(basal, rng.uniform(0.5, 1.5, (B, N)), -rng.uniform(0.1, 0.5, (B, N)))
        A = np.zeros((B, N, N))
        A[:, idx, idx] = np.where(basal, -rng.uniform(0.5, 1.0, (B, N)), -rng.uniform(0.05, 0.2, (B, N)))
        b = np.arange(B)
        for j in range(1, N):
            consumer = ~basal[:, j]
            forced = rng.integers(0, j, size=B)  # every consumer eats something
            for i in range(j):
                link = consumer & ((rng.random(B) < 0.5) | (forced == i))
                a = rng.uniform(0.5, 1.5, B)
                A[b, j, i] += np.where(link, 0.5 * a, 0)   # predator gains
                A[b, i, j] -= np.where(link, a, 0)         # prey loses
        adj = (np.abs(A) > 0).astype(float)
        adj[:, idx, idx] = 0
        adj = np.maximum(adj, adj.transpose(0, 2, 1))
        params = dict(r=r, A=A, sigma=rng.uniform(0.05, 0.2, (B, 1)), adj=adj)
        return rng.uniform(-1, 1, (B, N)), params

    def step(self, lx, p, rng):
        x = np.exp(lx)
        drift = p["r"] + np.einsum("bij,bj->bi", p["A"], x)
        lx = lx + drift * self.dt + p["sigma"] * np.sqrt(self.dt) * rng.standard_normal(lx.shape)
        return np.clip(lx, -9, 5)

    def obs(self, lx, p):
        return lx[..., None]


class Boids(Simulator):
    """Flock of 12 birds following cohesion / alignment / separation rules."""
    name, field, n, f, substeps = "boids", "animal behaviour", 12, 4, 2
    feature_names = ("x", "y", "vx", "vy")
    dt, L, R = 0.1, 10.0, 2.5

    def init(self, rng, B):
        N = self.n
        pos = rng.uniform(2, 8, (B, N, 2))
        ang = rng.uniform(0, 2 * np.pi, (B, N))
        vel = np.stack([np.cos(ang), np.sin(ang)], -1)
        params = dict(
            wc=rng.uniform(0.2, 1.0, (B, 1, 1)), wa=rng.uniform(0.2, 1.5, (B, 1, 1)),
            ws=rng.uniform(0.5, 2.0, (B, 1, 1)), sigma=rng.uniform(0.1, 0.5, (B, 1, 1)),
            adj=np.ones((B, N, N)) - np.eye(N)[None],
        )
        return (pos, vel), params

    def step(self, state, p, rng):
        pos, vel = state
        diff = pos[:, None, :, :] - pos[:, :, None, :]           # [B, i, j, 2]
        dist = np.linalg.norm(diff, axis=-1) + np.eye(self.n)[None] * 1e6
        nb = (dist < self.R).astype(float)
        cnt = nb.sum(-1, keepdims=True) + 1e-9
        has = (cnt > 0.5).astype(float)
        coh = (nb[..., None] * diff).sum(2) / cnt * has
        ali = ((nb[..., None] * vel[:, None]).sum(2) / cnt - vel) * has
        close = (dist < 1.0).astype(float)
        sep = -(close[..., None] * diff / (dist[..., None] ** 2 + 0.05)).sum(2)
        wall = 3.0 * (np.maximum(1.0 - pos, 0) - np.maximum(pos - (self.L - 1.0), 0))
        acc = p["wc"] * coh + p["wa"] * ali + p["ws"] * sep + wall
        acc = acc + p["sigma"] * rng.standard_normal(acc.shape) / np.sqrt(self.dt)
        vel = vel + acc * self.dt
        speed = np.linalg.norm(vel, axis=-1, keepdims=True) + 1e-9
        vel = vel / speed * np.clip(speed, 0.5, 2.0)
        return pos + vel * self.dt, vel

    def obs(self, state, p):
        pos, vel = state
        return np.concatenate([pos, vel], -1)


# ---------------------------------------------------------------- physics
class Kuramoto(Simulator):
    """12 coupled oscillators (fireflies, pacemaker cells, power grids)."""
    name, field, n, f, substeps = "kuramoto", "synchronisation", 12, 2, 4
    feature_names = ("cos phase", "sin phase")
    dt = 0.05

    def init(self, rng, B):
        N = self.n
        adj = _sym_er(rng, B, N, rng.uniform(0.2, 0.6, B))
        params = dict(
            omega=rng.standard_normal((B, N)) * rng.uniform(0.5, 1.5, (B, 1)),
            K=rng.uniform(0, 4, (B, 1)), sigma=rng.uniform(0.05, 0.3, (B, 1)),
            deg=np.maximum(adj.sum(-1), 1), adj=adj,
        )
        return rng.uniform(0, 2 * np.pi, (B, N)), params

    def step(self, th, p, rng):
        pull = (p["adj"] * np.sin(th[:, None, :] - th[:, :, None])).sum(-1) / p["deg"]
        dth = p["omega"] + p["K"] * pull
        return th + dth * self.dt + p["sigma"] * np.sqrt(self.dt) * rng.standard_normal(th.shape)

    def obs(self, th, p):
        return np.stack([np.cos(th), np.sin(th)], -1)


class Lorenz96(Simulator):
    """Chaotic ring of 10 variables, a standard toy model of the atmosphere."""
    name, field, n, f, substeps = "lorenz96", "climate / chaos", 10, 1, 5
    feature_names = ("x",)
    dt = 0.01

    def init(self, rng, B):
        N = self.n
        adj = np.zeros((B, N, N))
        for i in range(N):
            for d in (-2, -1, 1, 2):
                adj[:, i, (i + d) % N] = 1
        F = rng.uniform(4, 10, (B, 1))
        params = dict(F=F, sigma=rng.uniform(0.1, 0.5, (B, 1)), adj=adj)
        return F + rng.standard_normal((B, N)), params

    def step(self, x, p, rng):
        def f(y):
            return (np.roll(y, -1, 1) - np.roll(y, 2, 1)) * np.roll(y, 1, 1) - y + p["F"]
        h = self.dt
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x + p["sigma"] * np.sqrt(h) * rng.standard_normal(x.shape)

    def obs(self, x, p):
        return x[..., None]


class NBody(Simulator):
    """6 bodies orbiting under gravity (a tiny star cluster)."""
    name, field, n, f, substeps = "nbody", "astronomy", 6, 4, 10
    feature_names = ("x", "y", "vx", "vy")
    dt, eps = 0.01, 0.2
    max_burn = 20

    def init(self, rng, B):
        N = self.n
        m = rng.uniform(0.5, 2.0, (B, N))
        R = rng.uniform(1.0, 3.0, (B, 1))
        r = R * np.sqrt(rng.random((B, N)))
        a = rng.uniform(0, 2 * np.pi, (B, N))
        pos = np.stack([r * np.cos(a), r * np.sin(a)], -1)
        w = np.sqrt(m.sum(1, keepdims=True) / R ** 3) * rng.uniform(0.5, 0.9, (B, 1))
        vel = np.stack([-w * pos[..., 1], w * pos[..., 0]], -1) + 0.2 * rng.standard_normal((B, N, 2))
        mt = m.sum(1)[:, None, None]
        pos -= (m[..., None] * pos).sum(1, keepdims=True) / mt
        vel -= (m[..., None] * vel).sum(1, keepdims=True) / mt
        params = dict(m=m, adj=np.ones((B, N, N)) - np.eye(N)[None])
        return (pos, vel), params

    def _acc(self, pos, m):
        diff = pos[:, None, :, :] - pos[:, :, None, :]
        d3 = (np.sum(diff ** 2, -1) + self.eps ** 2) ** 1.5
        return (m[:, None, :, None] * diff / d3[..., None]).sum(2)

    def step(self, state, p, rng):
        pos, vel = state
        vel = vel + 0.5 * self.dt * self._acc(pos, p["m"])
        pos = pos + self.dt * vel
        vel = vel + 0.5 * self.dt * self._acc(pos, p["m"])
        vel = vel + 0.01 * np.sqrt(self.dt) * rng.standard_normal(vel.shape)
        return np.clip(pos, -20, 20), vel

    def obs(self, state, p):
        pos, vel = state
        return np.concatenate([pos, vel], -1)


# ---------------------------------------------------------------- biology
class GeneRegulation(Simulator):
    """8 genes switching each other on and off (a gene regulatory network)."""
    name, field, n, f, substeps = "gene_reg", "cell biology", 8, 1, 4
    feature_names = ("expression",)
    dt = 0.05

    def init(self, rng, B):
        N = self.n
        links = rng.random((B, N, N)) < 0.3
        W = links * rng.choice([-1.0, 1.0], (B, N, N)) * rng.uniform(1, 4, (B, N, N))
        adj = (links | links.transpose(0, 2, 1)).astype(float)
        adj[:, np.arange(N), np.arange(N)] = 0
        params = dict(
            W=W, theta=rng.uniform(-1, 2, (B, N)), alpha=rng.uniform(1, 3, (B, N)),
            delta=rng.uniform(0.5, 1.5, (B, N)), sigma=rng.uniform(0.05, 0.2, (B, 1)), adj=adj,
        )
        return rng.uniform(0, 2, (B, N)), params

    def step(self, x, p, rng):
        drive = np.einsum("bij,bj->bi", p["W"], x) - p["theta"]
        prod = 0.05 + p["alpha"] / (1 + np.exp(-drive))
        dx = prod - p["delta"] * x
        noise = p["sigma"] * np.sqrt(x + 0.01) * np.sqrt(self.dt) * rng.standard_normal(x.shape)
        return np.maximum(x + dx * self.dt + noise, 0)

    def obs(self, x, p):
        return x[..., None]


class Neurons(Simulator):
    """10 coupled FitzHugh-Nagumo neurons (spiking brain-circuit model)."""
    name, field, n, f, substeps = "neurons", "neuroscience", 10, 2, 8
    feature_names = ("voltage", "recovery")
    dt = 0.2

    def init(self, rng, B):
        N = self.n
        adj = _sym_er(rng, B, N, 0.3)
        params = dict(
            I=rng.uniform(0, 1, (B, N)), g=rng.uniform(0, 1, (B, 1)),
            sigma=rng.uniform(0.05, 0.3, (B, 1)), deg=np.maximum(adj.sum(-1), 1), adj=adj,
        )
        return (rng.uniform(-2, 2, (B, N)), rng.uniform(-1, 1, (B, N))), params

    def step(self, state, p, rng):
        v, w = state
        coup = (p["adj"] * (v[:, None, :] - v[:, :, None])).sum(-1) / p["deg"]
        dv = v - v ** 3 / 3 - w + p["I"] + p["g"] * coup
        dw = 0.08 * (v + 0.7 - 0.8 * w)
        v = v + dv * self.dt + p["sigma"] * np.sqrt(self.dt) * rng.standard_normal(v.shape)
        return v, w + dw * self.dt

    def obs(self, state, p):
        return np.stack(state, -1)


# ---------------------------------------------------------------- society
class Epidemic(Simulator):
    """Disease spreading between 10 linked cities (stochastic SIR)."""
    name, field, n, f, substeps = "epidemic", "society / epidemiology", 10, 3, 2
    feature_names = ("susceptible", "infected", "recovered")
    dt = 0.5
    event_f = 1

    def init(self, rng, B):
        N = self.n
        pop = rng.integers(2000, 20000, (B, N))
        I = np.zeros((B, N), dtype=np.int64)
        I[np.arange(B), rng.integers(0, N, B)] = rng.integers(1, 10, B)
        adj = _sym_er(rng, B, N, 0.3)
        params = dict(
            pop=pop, beta=rng.uniform(0.2, 0.6, (B, 1)), gamma=rng.uniform(0.1, 0.2, (B, 1)),
            m=rng.uniform(0.01, 0.2, (B, 1)), deg=np.maximum(adj.sum(-1), 1), adj=adj,
        )
        return (pop - I, I, np.zeros_like(I)), params

    def step(self, state, p, rng):
        S, I, R = state
        frac = I / p["pop"]
        nbr = (p["adj"] * frac[:, None, :]).sum(-1) / p["deg"]
        lam = p["beta"] * ((1 - p["m"]) * frac + p["m"] * nbr)
        new_inf = rng.binomial(S, 1 - np.exp(-lam * self.dt))
        new_rec = rng.binomial(I, 1 - np.exp(-p["gamma"] * self.dt))
        return S - new_inf, I + new_inf - new_rec, R + new_rec

    def obs(self, state, p):
        return np.stack(state, -1) / p["pop"][..., None]


SIMULATORS = {s.name: s for s in [
    PredatorPrey(), Boids(), Kuramoto(), Lorenz96(), NBody(), GeneRegulation(), Neurons(), Epidemic(),
]}
DOMAINS = list(SIMULATORS)


def repeat_tree(tree, k):
    """Copy every episode k times (used to branch one history into many futures)."""
    if isinstance(tree, dict):
        return {key: repeat_tree(v, k) for key, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(repeat_tree(v, k) for v in tree)
    return np.repeat(tree, k, axis=0)
