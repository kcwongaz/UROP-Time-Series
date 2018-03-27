import numpy as np


# General Tools
def prob_bin(p):

    """Binary random variable"""
    if np.random.uniform() < p:
        return True
    else:
        return False


# Signal processing functions

def gen_ts_m(w1, w2, tm, t_max, a1=1, a2=1, phi1=0, phi2=0, init=1):

    """Generate signal from hidden Markov, 1 is fast, 2 is slow"""

    state = []
    x = []

    if phi1 == 0:
        phi1 = np.random.uniform(0, 2*np.pi)

    if phi2 == 0:
        phi2 = np.random.uniform(0, 2*np.pi)

    if prob_bin(init):
        state.append("F")
    else:
        state.append("S")

    for t in range(0, t_max):

        if state[t] == 1:
            x.append(a1 * np.sin(w1*t + phi1))

            if prob_bin(tm["F"]["F"]):
                state.append("F")
            else:
                state.append("S")

        else:
            x.append(a2 * np.sin(w2*t + phi2))

            if prob_bin(tm["S"]["F"]):
                state.append("F")
            else:
                state.append("S")

    return np.array(x), np.array(state)


def gen_ts_h(w, v, t_max, a=1, b=1):

    """Generate signal of Hilbert-Haung type"""

    taxis = np.arange(0, t_max)
    return np.sin(w * np.sin(v * taxis + b) + a)


def discretize(x, small=0.1, large=0.2):

    """Discretize input into the spin-2 chain of length (t_max - 1)"""

    t_max = len(x)

    s = []
    for t in range(1, t_max):

        delta = (x[t] - x[t-1]) / x[t-1]

        if delta > large:
            s.append(2)
        elif delta > small:
            s.append(1)
        elif delta < -large:
            s.append(-2)
        elif delta < -small:
            s.append(-1)
        else:
            s.append(0)

    return s


def emission(x, delta, small=0.1, large=0.2):

    """Discretize input into the emission states"""

    t_max = len(x)
    n_max = int(t_max / delta) - 1
    s = []

    for n in range(n_max):

        diff = 0
        for i in range(0, delta):

            diff += (x[n*delta + i + 1] - x[n*delta + i]) / x[n*delta + i]

            if abs(diff) > large:
                s.append(2)
            elif abs(diff) > small:
                s.append(1)
            else:
                s.append(0)

    return s


def emission_fft(x, small=0.1, large=0.2):

    """Discretize input by FFT and cumulative difference"""

    period = 1 / (len(x)-1)
    delta = period / np.argmax(np.fft.fft(x))

    return emission(x, delta, small=small, large=large)


# Pattern recognition for transition probabilities

def normalize(prob):

    """Normalize the given set of transition probabilities"""

    const = 0
    for p in prob:
        const += p

    if const == 0:
        const = 1

    for p in prob:
        p /= const

    return prob


def trans_a(a, w, t_max):

    """Construct the transition matrix for A"""

    tm = {"UU": 0.0, "UD": 0.0, "DU": 0.0, "DD": 0.0}
    x = a * np.sin(w * np.arange(0, t_max))

    for t in range(2, t_max):

        if x[t-1] - x[t-2] > 0:
            if x[t] - x[t-1] > 0:
                tm["UU"] += 1
            else:
                tm["UD"] += 1

        else:
            if x[t] - x[t-1] > 0:
                tm["DU"] += 1
            else:
                tm["DD"] += 1

    tm["UU"], tm["UD"] = normalize([tm["UU"], tm["UD"]])
    tm["DU"], tm["DD"] = normalize([tm["DU"], tm["DD"]])

    return tm


def trans_b(a, w, t_max, large=0.2, small=0.1):

    """Construct the transition matrix for B"""
    tm = {
        "GU": 0.0,
        "GD": 0.0,
        "WU": 0.0,
        "WD": 0.0,
        "HU": 0.0,
        "HD": 0.0,
        "LU": 0.0,
        "LD": 0.0,
        "CU": 0.0,
        "CD": 0.0
        }

    x = a * np.sin(w * np.arange(0, t_max))

    for t in range(2, t_max):
        frac = (x[t-1] - x[t-2]) / x[t-2]

        # Large gain
        if frac > large:
            if x[t] - x[t-1] > 0:
                tm["GU"] += 1
            else:
                tm["GD"] += 1

        # Small win
        elif frac > small:
            if x[t] - x[t-1] > 0:
                tm["WU"] += 1
            else:
                tm["WD"] += 1

        # Large crash
        elif frac < -large:
            if x[t] - x[t-1] > 0:
                tm["CU"] += 1
            else:
                tm["CD"] += 1

        # Small Loss
        elif frac < -small:
            if x[t] - x[t-1] > 0:
                tm["LU"] += 1
            else:
                tm["LD"] += 1

        # Hold
        else:
            if x[t] - x[t - 1] > 0:
                tm["HU"] += 1
            else:
                tm["HD"] += 1

    tm["GU"], tm["GD"] = normalize([tm["GU"], tm["GD"]])
    tm["WU"], tm["WD"] = normalize([tm["WU"], tm["WD"]])
    tm["CU"], tm["CD"] = normalize([tm["CU"], tm["CD"]])
    tm["LU"], tm["LD"] = normalize([tm["LU"], tm["LD"]])
    tm["HU"], tm["HD"] = normalize([tm["HU"], tm["HD"]])

    return tm


def trans_a_dir(x, s, delta):

    """Construct the transition matrices directly from the discretized signal"""


def trans_emit_inv(w, t_max, delta):

    """Compute the emission probability for a single component"""

    trial = 100

    tm = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        }

    s = emission(np.sin(w * np.arange(0, t_max) + np.random.uniform(0, 2 * np.pi)), delta)
    for _ in range(trial):

        for t in range(len(s)):

            if s[t] == 0:
                tm[0] += 1
            elif s[t] == 1:
                tm[1] += 1
            else:
                tm[2] += 1

            s = emission(np.sin(w * np.arange(0, t_max) + np.random.uniform(0, 2 * np.pi)), delta)

    for i in range(3):
        tm[i] /= len(s) * trial

    return tm


def trans_emit(w1, w2, t_max, delta=1):

    tm_f = trans_emit_inv(w1, t_max, delta)
    tm_s = trans_emit_inv(w2, t_max, delta)

    tm = {"F": {0: tm_f[0], 1: tm_f[1], 2: tm_f[2]},
          "S": {0: tm_s[0], 1: tm_s[1], 2: tm_s[2]}}

    return tm


# Learning algorithm

def viterbi(x, init_p, trans_p, emit_p, delta=1):

    """Viterbi algorithm"""

    state = ("F", "S")

    obs = emission(x, delta)
    t_max = len(obs)
    v = [{}]

    # Initial step
    for si in state:
        v[0][si] = {"prob": init_p[0] * emit_p[si][obs[0]], "prev": None}

    # Forward stage
    for t in range(1, t_max):
        v.append({})

        for si in state:

            prob_max = -1
            arg_max = None

            for sj in state:
                p = emit_p[si][obs[t]] * trans_p[sj][si] * v[t-1][sj]["prob"]

                if p > prob_max:
                    prob_max = p
                    arg_max = sj

            v[t][si] = {"prob": prob_max, "prev": arg_max}

    # Backward stage, obtain the Viterbi path
    v_path = []
    prev = None

    ## May still have bug
    #
    # prob_max = 0
    #
    # for si in state:
    #     p = v[-1][si]["prob"]
    #
    #     if p > prob_max:
    #         prob_max = p
    #         v_path.append(si)
    #         prev = v[-1][si]["prev"]
    #
    # for t in range(t_max-2, -1, -1):
    #
    #     v_path.append(prev)
    #     prev = v[t][prev]["prev"]
    ##

    ## Copied from wiki
    prob_max = max(value["prob"] for value in v[-1].values())

    # Get most probable state and its backtrack
    for st, data in v[-1].items():
        if data["prob"] == prob_max:
            v_path.append(st)
            prev = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(v) - 2, -1, -1):
        v_path.insert(0, v[t + 1][prev]["prev"])
        prev = v[t + 1][prev]["prev"]

    return v_path, prob_max


def filter_alg():

    """Filter algorithm"""


# Prediction functions

def seq_sto(gamma, t_max, sym1="A", sym2="B"):

    """Create a stochastic sequence of symbols"""

    seq = []
    for t in range(t_max - 2):  # The AB seq has length 2 shorter than x(t)

        if prob_bin(gamma) < gamma:
            seq.append(sym1)
        else:
            seq.append(sym2)

    return seq


def eval_seq(seq, x, tm_a, tm_b, trial, small=0.1, large=0.2):

    eff = 0
    t_max = len(x)

    for _ in range(trial):

        for t in range(2, t_max):

            """(1) the seq is 2 unit shorter than the time series
               (2) we are making prediction at every t to predict x[t]
            """

            delta = (x[t-1] - x[t-2]) / x[t-2]
            state = x[t] - x[t-1]
            rand = np.random.uniform()

            if seq[t-2] == "A":
                if delta > 0:
                    if rand < tm_a["UU"] and state > 0:
                        eff += 1
                    elif rand > tm_a["UU"] and state < 0:
                        eff += 1

                else:
                    if rand < tm_a["DU"] and state > 0:
                        eff += 1
                    elif rand > tm_a["DU"] and state < 0:
                        eff += 1

            elif seq[t-2] == "B":
                if delta > large:
                    if rand < tm_b["GU"] and state > 0:
                        eff += 1
                    elif rand > tm_b["GU"] and state < 0:
                        eff += 1

                elif delta > small:
                    if rand < tm_b["WU"] and state > 0:
                        eff += 1
                    elif rand > tm_b["WU"] and state < 0:
                        eff += 1

                elif delta < -large:
                    if rand < tm_b["CU"] and state > 0:
                        eff += 1
                    elif rand > tm_b["CU"] and state < 0:
                        eff += 1

                elif delta < -small:
                    if rand < tm_b["LU"] and state > 0:
                        eff += 1
                    elif rand > tm_b["LU"] and state < 0:
                        eff += 1

                else:
                    if rand < tm_b["HU"] and state > 0:
                        eff += 1
                    elif rand > tm_b["HU"] and state < 0:
                        eff += 1

    return eff / (trial * t_max)
