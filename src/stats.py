"""Interval estimation and the four hypothesis tests.

Every interval in the paper comes from one procedure: a cluster bootstrap that
resamples ITEMS with replacement, keeping all of an item's conversations
together, 2,000 resamples, percentile intervals, seed 0. Conversations on the
same item are not independent, so resampling conversations would understate the
uncertainty.

Where a comparison is paired, meaning the same items appear in both conditions,
the items are resampled ONCE and the draw is applied to both. Resampling the two
conditions separately would discard the pairing and inflate the interval.

One-sided bootstrap p-values are the proportion of resamples falling on the side
of zero opposite to the prediction, with the conventional (k + 1) / (B + 1)
correction so that p is never exactly zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

N_BOOT = 2000
SEED = 0


# ---------------------------------------------------------------- machinery

def _item_agg(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """items, conversations per item, successes per item. One pass, so the
    bootstrap costs O(n_items) per draw rather than a dataframe rebuild."""
    g = df.groupby("item")[col].agg(["size", "sum"])
    return g.index.to_numpy(), g["size"].to_numpy(float), g["sum"].to_numpy(float)


def _draw(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    return rng.integers(0, n, size=(b, n))


def boot_rate(df: pd.DataFrame, col: str = "flip", n_boot: int = N_BOOT,
              seed: int = SEED) -> np.ndarray:
    if df.empty:
        return np.array([])
    _, size, succ = _item_agg(df, col)
    idx = _draw(np.random.default_rng(seed), len(size), n_boot)
    return succ[idx].sum(1) / size[idx].sum(1)


def ci(samples: np.ndarray) -> tuple[float, float]:
    if len(samples) == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def p_one_sided(samples: np.ndarray, direction: str) -> float:
    """direction 'neg': H predicts the difference is negative, so p is the share
    of resamples at or above zero. 'pos' is the mirror image."""
    if len(samples) == 0:
        return np.nan
    k = (samples >= 0).sum() if direction == "neg" else (samples <= 0).sum()
    return float((k + 1) / (len(samples) + 1))


def dprime_c_arr(hits: np.ndarray, n_b: np.ndarray,
                 fas: np.ndarray, n_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised log-linear correction, as in analyse.dprime_c."""
    h = (hits + 0.5) / (n_b + 1)
    f = (fas + 0.5) / (n_a + 1)
    zh, zf = norm.ppf(h), norm.ppf(f)
    return zh - zf, -0.5 * (zh + zf)


def boot_sdt(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = N_BOOT,
             seed: int = SEED) -> dict[str, np.ndarray]:
    """Set A and Set B hold different items by construction, so they are
    resampled independently, from one generator for reproducibility."""
    if a.empty or b.empty:
        return {k: np.array([]) for k in ("H", "F", "d", "c")}
    rng = np.random.default_rng(seed)
    _, sa, ka = _item_agg(a, "flip")
    _, sb, kb = _item_agg(b, "flip")
    ia, ib = _draw(rng, len(sa), n_boot), _draw(rng, len(sb), n_boot)
    n_a, fas = sa[ia].sum(1), ka[ia].sum(1)
    n_b, hits = sb[ib].sum(1), kb[ib].sum(1)
    d, c = dprime_c_arr(hits, n_b, fas, n_a)
    return {"H": hits / n_b, "F": fas / n_a, "d": d, "c": c}


def boot_sdt_paired(cells: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
                    n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, dict[str, np.ndarray]]:
    """Several conditions over the SAME items: resample items once, apply to all.

    Used for the mitigation comparison, where a replay reruns the identical
    items under a different system prompt. Independent resampling would throw
    the pairing away.
    """
    names = list(cells)
    items_a = sorted(set.intersection(*[set(cells[n][0]["item"]) for n in names]))
    items_b = sorted(set.intersection(*[set(cells[n][1]["item"]) for n in names]))
    if not items_a or not items_b:
        return {}
    rng = np.random.default_rng(seed)
    ia, ib = _draw(rng, len(items_a), n_boot), _draw(rng, len(items_b), n_boot)
    out = {}
    for n in names:
        a = cells[n][0][cells[n][0]["item"].isin(items_a)]
        b = cells[n][1][cells[n][1]["item"].isin(items_b)]
        ga = a.groupby("item")["flip"].agg(["size", "sum"]).reindex(items_a).fillna(0)
        gb = b.groupby("item")["flip"].agg(["size", "sum"]).reindex(items_b).fillna(0)
        sa, ka = ga["size"].to_numpy(float), ga["sum"].to_numpy(float)
        sb, kb = gb["size"].to_numpy(float), gb["sum"].to_numpy(float)
        n_a, fas = sa[ia].sum(1), ka[ia].sum(1)
        n_b, hits = sb[ib].sum(1), kb[ib].sum(1)
        d, c = dprime_c_arr(hits, n_b, fas, n_a)
        out[n] = {"H": hits / n_b, "F": fas / n_a, "d": d, "c": c}
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni, applied WITHIN a model across its hypothesis tests."""
    items = sorted([(p, k) for k, p in pvals.items() if np.isfinite(p)])
    m = len(items)
    adj, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    for k, p in pvals.items():
        adj.setdefault(k, np.nan)
    return adj


# ---------------------------------------------------------------- hypotheses

def h1_citation_vs_plain(df: pd.DataFrame) -> pd.DataFrame:
    """H1: citation-shaped pressure shifts the criterion without raising d'.

    The comparison is citation against PLAIN disagreement. Plain statements were
    run inside the identity condition, whose "none" sub-condition supplies no
    claimed identity, so plain disagreement is identity with identity == None.
    Pooling all four identity sub-conditions would compare citation against
    identity-framed disagreement instead, which is a different contrast and not
    the one the hypothesis states.

    Both conditions use the same items, so the draw is shared.
    Prediction: c is LOWER under citation, meaning readier concession.
    """
    rows = []
    d1 = df[df.arm == "ascending"].copy()
    # The no-identity sub-condition may be stored as a null or as the string
    # "none", depending on how the condition list was parsed from the config, so
    # accept both. Getting this wrong silently empties the table.
    ident = d1["identity"]
    is_plain = ident.isna() | ident.astype(str).str.strip().str.lower().isin(
        {"none", "nan", ""})
    plain = d1[(d1.type == "identity") & is_plain]
    if plain.empty and not d1[d1.type == "identity"].empty:
        raise SystemExit(
            "H1: no plain rows found. The identity values present are "
            f"{sorted(set(d1[d1.type=='identity'].identity.astype(str)))}. "
            "Update the no-identity test in h1_citation_vs_plain.")
    for (model, system), g in d1.groupby(["model", "system"]):
        cells = {}
        gp = plain[(plain.model == model) & (plain.system == system)]
        for t in ("citation", "plain"):
            src = g[g.type == "citation"] if t == "citation" else gp
            a = src[src.set == "A"]
            b = src[src.set == "B"]
            if a.empty or b.empty:
                break
            cells[t] = (a, b)
        if len(cells) != 2:
            continue
        s = boot_sdt_paired(cells)
        if not s:
            continue
        dc = s["citation"]["c"] - s["plain"]["c"]
        dd = s["citation"]["d"] - s["plain"]["d"]
        rows.append({
            "model": model, "system": system,
            "delta_c": float(dc.mean()), "c_lo": ci(dc)[0], "c_hi": ci(dc)[1],
            "p_c_one_sided_neg": p_one_sided(dc, "neg"),
            "delta_d": float(dd.mean()), "d_lo": ci(dd)[0], "d_hi": ci(dd)[1],
            "comparison": "citation vs plain (identity sub-condition with no identity)",
            "n_A_plain": len(cells["plain"][0]), "n_A_citation": len(cells["citation"][0]),
        })
    return pd.DataFrame(rows)


def h4_mitigation(df: pd.DataFrame) -> pd.DataFrame:
    """H4: the evidence gate raises d' against both reference prompts.

    Paired: the replay reruns identical items, so items are resampled once and
    the same draw is applied to every condition.
    """
    rows = []
    d1 = df[df.arm == "ascending"]
    for (model, ptype), g in d1.groupby(["model", "type"]):
        systems = set(g.system.unique())
        if "evidence_gate" not in systems:
            continue
        for ref in ("baseline", "generic_warning"):
            if ref not in systems:
                continue
            cells = {}
            for s in ("evidence_gate", ref):
                cells[s] = (g[(g.system == s) & (g.set == "A")],
                            g[(g.system == s) & (g.set == "B")])
            if any(x.empty for pair in cells.values() for x in pair):
                continue
            bs = boot_sdt_paired(cells)
            if not bs:
                continue
            dd = bs["evidence_gate"]["d"] - bs[ref]["d"]
            dcc = bs["evidence_gate"]["c"] - bs[ref]["c"]
            rows.append({
                "model": model, "type": ptype, "reference": ref,
                "delta_d": float(dd.mean()), "d_lo": ci(dd)[0], "d_hi": ci(dd)[1],
                "p_d_one_sided_pos": p_one_sided(dd, "pos"),
                "delta_c": float(dcc.mean()), "c_lo": ci(dcc)[0], "c_hi": ci(dcc)[1],
            })
    return pd.DataFrame(rows)


def h3_denial(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H3: denial of influence is above zero and differs by model.

    Above zero: the bootstrap interval for the rate, resampled over ITEMS.
    Differs by model: a chi-square test of denial against model, computed on
    Set A, with conversation-level counts. The chi-square ignores item
    clustering, so pairwise bootstrap intervals for the differences are given
    alongside and are the more conservative evidence.
    """
    import json
    import os
    if not os.path.exists(path):
        return pd.DataFrame(), pd.DataFrame()
    rows = [r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())
            if r.get("influence")]
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    d = pd.DataFrame(rows).rename(columns={"item_id": "item", "system_prompt": "system"})
    d["denial"] = (d.influence == "no").astype(int)

    per = []
    for (model, system, st), g in d.groupby(["model", "system", "set"]):
        s = boot_rate(g, "denial")
        lo, hi = ci(s)
        per.append({"model": model, "system": system, "set": st, "n": len(g),
                    "denial_rate": g.denial.mean(), "lo": lo, "hi": hi,
                    "above_zero": lo > 0})

    tests = []
    for (system, st), g in d.groupby(["system", "set"]):
        tab = pd.crosstab(g.model, g.denial)
        if tab.shape[0] > 1 and tab.shape[1] > 1:
            chi2, p, dof, _ = chi2_contingency(tab)
            tests.append({"system": system, "set": st, "test": "chi-square, denial by model",
                          "chi2": chi2, "dof": dof, "p": p,
                          "note": "conversation level, ignores item clustering"})
    return pd.DataFrame(per), pd.DataFrame(tests)


def neutral_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Each pressure type against the neutral control, per set.

    Same items in both, so the draw is shared.
    """
    rows = []
    d1 = df[df.arm == "ascending"]
    for (model, system, st), g in d1.groupby(["model", "system", "set"]):
        ctrl = g[g.type == "neutral"]
        if ctrl.empty:
            continue
        for t in sorted(set(g.type.unique()) - {"neutral"}):
            sub = g[g.type == t]
            if sub.empty:
                continue
            items = sorted(set(sub.item) & set(ctrl.item))
            if not items:
                continue
            rng = np.random.default_rng(SEED)
            idx = _draw(rng, len(items), N_BOOT)
            gt = sub[sub.item.isin(items)].groupby("item")["flip"].agg(
                ["size", "sum"]).reindex(items).fillna(0)
            gc = ctrl[ctrl.item.isin(items)].groupby("item")["flip"].agg(
                ["size", "sum"]).reindex(items).fillna(0)
            rt = gt["sum"].to_numpy(float)[idx].sum(1) / gt["size"].to_numpy(float)[idx].sum(1)
            rc = gc["sum"].to_numpy(float)[idx].sum(1) / gc["size"].to_numpy(float)[idx].sum(1)
            diff = rt - rc
            lo, hi = ci(diff)
            rows.append({"model": model, "system": system, "set": st, "type": t,
                         "rate": sub.flip.mean(), "neutral_rate": ctrl.flip.mean(),
                         "delta": float(diff.mean()), "lo": lo, "hi": hi,
                         "p_one_sided_pos": p_one_sided(diff, "pos")})
    return pd.DataFrame(rows)


def dose_response_single_shot(df: pd.DataFrame) -> pd.DataFrame:
    """H2: concession on mean rated strength, single-shot arm, both sets.

    The single-shot arm presents each tier alone, so strength is not confounded
    with accumulated pressure. Set enters as a covariate. Standard errors are
    clustered by item; a mixed-effects version with a random intercept per item
    was not feasible in the available time and is noted as a limitation.

    Separation is flagged rather than reported as an estimate. A coefficient of
    large magnitude with a p-value near one means the likelihood did not
    converge, not a large effect.
    """
    import statsmodels.formula.api as smf

    d = df[df.arm.astype(str).str.startswith("single_shot") & df.flip_rating.notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["set_b"] = (d["set"] == "B").astype(int)
    rows = []
    for (model, ptype), g in d.groupby(["model", "type"]):
        n = len(g)
        if g.flip.nunique() < 2 or n < 30:
            rows.append({"model": model, "type": ptype, "n": n, "status": "too few or no variation"})
            continue
        try:
            m = smf.logit("flip ~ flip_rating + set_b", data=g).fit(
                disp=False, cov_type="cluster", cov_kwds={"groups": g["item"]})
            beta = float(m.params["flip_rating"])
            p_two = float(m.pvalues["flip_rating"])
            # H2 predicts stronger statements produce MORE concession, so the
            # test is one-sided on beta > 0. statsmodels reports two-sided.
            p = p_two / 2 if beta > 0 else 1 - p_two / 2
            fitted = m.predict(g)
            separated = (abs(beta) > 5) or (p > 0.99) or bool(
                ((fitted < 1e-6) | (fitted > 1 - 1e-6)).any()) or not m.mle_retvals.get(
                "converged", True)
            rows.append({
                "model": model, "type": ptype, "n": n,
                "beta_strength": beta, "se": float(m.bse["flip_rating"]),
                "p_one_sided_pos": p, "p_two_sided": p_two,
                "beta_set_b": float(m.params["set_b"]),
                "status": "SEPARATION, do not interpret" if separated else "ok",
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": model, "type": ptype, "n": n, "status": f"failed: {exc}"})
    return pd.DataFrame(rows)


def dose_response_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """One H2 test per model, pooling pressure types.

    Taking the smallest p across a model's pressure types would be selection
    without correction, so H2 is instead a single fit per model with the type as
    a fixed effect, the set as a covariate and standard errors clustered by item.
    That gives one coefficient for rated strength per model and one one-sided p,
    which is what enters the Holm family. The per-type table is retained for
    description only.
    """
    import statsmodels.formula.api as smf

    d = df[df.arm.astype(str).str.startswith("single_shot") & df.flip_rating.notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["set_b"] = (d["set"] == "B").astype(int)
    rows = []
    for model, g in d.groupby("model"):
        if g.flip.nunique() < 2 or len(g) < 50 or g.type.nunique() < 2:
            rows.append({"model": model, "n": len(g), "status": "too few or no variation"})
            continue
        # Drop any covariate that does not vary in this model's data, otherwise
        # the design matrix is singular and the fit fails outright.
        terms = ["flip_rating"]
        if g.set_b.nunique() > 1:
            terms.append("set_b")
        if g.type.nunique() > 1:
            terms.append("C(type)")
        formula = "flip ~ " + " + ".join(terms)
        try:
            m = smf.logit(formula, data=g).fit(
                disp=False, cov_type="cluster", cov_kwds={"groups": g["item"]})
            beta = float(m.params["flip_rating"])
            p_two = float(m.pvalues["flip_rating"])
            p = p_two / 2 if beta > 0 else 1 - p_two / 2
            fitted = m.predict(g)
            sep = (abs(beta) > 5) or bool(((fitted < 1e-6) | (fitted > 1 - 1e-6)).any()) \
                or not m.mle_retvals.get("converged", True)
            rows.append({"model": model, "n": len(g), "n_types": int(g.type.nunique()),
                         "formula": formula,
                         "beta_strength": beta, "se": float(m.bse["flip_rating"]),
                         "p_one_sided_pos": p, "p_two_sided": p_two,
                         "status": "SEPARATION, do not interpret" if sep else "ok"})
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": model, "n": len(g), "status": f"failed: {exc}"})
    return pd.DataFrame(rows)


def holm_table(h1: pd.DataFrame, h3_tests: pd.DataFrame, h4: pd.DataFrame,
               h2: pd.DataFrame) -> pd.DataFrame:
    """Holm correction applied WITHIN each model.

    The family is the hypothesis tests that are estimated separately for each
    model, which are H1, H2 and H4. The correction is not applied across models,
    since each model is a separate question.

    H3's model comparison is excluded by design: it is a single omnibus test
    spanning all models, so it has no per-model version, and including it would
    inflate the correction for the tests that do. `h3_tests` is accepted for the
    signature's stability and to record that the omission is deliberate.
    """
    p = {}
    if not h1.empty:
        # H1 is pre-specified on the BASELINE condition. Iterating every system
        # would let the last row written win, which previously meant a
        # mitigation condition silently supplied the model's H1 p.
        for _, r in h1[h1.system == "baseline"].iterrows():
            p.setdefault(r.model, {})["H1 citation vs plain, c"] = r.p_c_one_sided_neg
    if not h2.empty and "p_one_sided_pos" in h2.columns:
        # One pooled fit per model, so no selection across pressure types.
        for _, r in h2[h2.get("status", "ok") == "ok"].iterrows():
            p.setdefault(r.model, {})["H2 strength"] = float(r.p_one_sided_pos)
    if not h4.empty:
        # H4 claims the gate beats BOTH reference prompts, so for each pressure
        # type the evidence is the LARGER of the two reference p-values, an
        # intersection-union test. Across types the smallest such value is taken
        # and Bonferroni-corrected by the number of types tested, since choosing
        # the best type is itself a selection.
        for m, g in h4.groupby("model"):
            per_type = g.groupby("type").p_d_one_sided_pos.max()
            if per_type.empty:
                continue
            k = len(per_type)
            p.setdefault(m, {})["H4 gate raises d'"] = float(min(1.0, per_type.min() * k))
    rows = []
    for model, tests in p.items():
        adj = holm(tests)
        for k, raw in tests.items():
            rows.append({"model": model, "test": k, "p_raw": raw, "p_holm": adj[k],
                         "family": f"{len(tests)} per-model tests ("
                                   + ", ".join(sorted(t.split()[0] for t in tests))
                                   + "); H3's omnibus comparison across models is "
                                     "reported separately, uncorrected"})
    return pd.DataFrame(rows)