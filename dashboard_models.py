"""Profile methods shared with the executed notebook; no coordinates enter fitting."""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import warnings

GMM_SEED = 42
GMM_COMPONENTS = range(1, 6)
PROFILE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#332288"]
DEPTH_LABELS = ["0–<70 km", "70–<300 km", "≥300 km"]

def prepare_profiles(frame):
    selected = frame.copy()
    audit = [{"step": "Cleaned catalog", "retained": len(selected), "excluded_at_step": 0}]
    rules = [
        ("Valid coordinates", lambda d: np.isfinite(d.latitude) & np.isfinite(d.longitude) & d.latitude.between(-90,90) & d.longitude.between(-180,180)),
        ("Finite positive magnitude", lambda d: np.isfinite(d.mag) & d.mag.gt(0)),
        ("Finite nonnegative depth", lambda d: np.isfinite(d.depth) & d.depth.ge(0)),
        ("Reported magnitude type", lambda d: d.magType.notna() & d.magType.astype(str).str.strip().ne("")),
    ]
    for name, rule in rules:
        previous = len(selected)
        selected = selected.loc[rule(selected)].copy()
        audit.append({"step": name, "retained": len(selected), "excluded_at_step": previous-len(selected)})
    counts = selected.magType.value_counts()
    common = sorted(counts.index[counts.eq(counts.max())])[0] if len(counts) else None
    previous = len(selected)
    selected = selected.loc[selected.magType.eq(common)].copy() if common is not None else selected.iloc[:0].copy()
    audit.append({"step": f"Most common eligible magnitude type: {common}", "retained": len(selected), "excluded_at_step": previous-len(selected)})
    audit = pd.DataFrame(audit)
    assert audit.excluded_at_step.sum() + len(selected) == len(frame)
    return selected, audit, common

def profile_features(frame, log_depth=True):
    # Geographic information is deliberately absent from the fitted features.
    values = frame[["mag", "depth"]].to_numpy(dtype=float).copy()
    if log_depth:
        values[:, 1] = np.log1p(values[:, 1])
    scaler = StandardScaler()
    return scaler.fit_transform(values), scaler

def fit_profiles(frame, variant, log_depth=True):
    rows, fitted = [], {}
    if len(frame) < 20 or frame[["mag","depth"]].nunique().lt(2).any():
        return pd.DataFrame([{"variant":variant,"components":0,"n_events":len(frame),"valid":False,"status":"Insufficient observations or feature variation"}]), None, None
    X, scaler = profile_features(frame, log_depth)
    for k in GMM_COMPONENTS:
        row = {"variant":variant,"components":k,"n_events":len(frame),"valid":False}
        if len(np.unique(X,axis=0)) < k:
            row["status"] = "Fewer distinct feature pairs than components"
            rows.append(row)
            continue
        try:
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always", ConvergenceWarning)
                model = GaussianMixture(n_components=k,covariance_type="full",n_init=10,
                    random_state=GMM_SEED,reg_covar=1e-4,max_iter=500).fit(X)
            bic = model.bic(X)
            valid = bool(model.converged_ and np.isfinite(bic))
            row.update(bic=bic,converged=bool(model.converged_),iterations=int(model.n_iter_),
                       valid=valid,status="Converged" if valid else "Nonconverged or nonfinite BIC",
                       warnings="; ".join(str(w.message) for w in caught))
            if valid: fitted[k] = model
        except (ValueError, np.linalg.LinAlgError) as error:
            row["status"] = str(error)
        rows.append(row)
    table = pd.DataFrame(rows)
    valid_rows = table.loc[table.valid]
    if valid_rows.empty: return table, None, scaler
    best_k = int(valid_rows.sort_values(["bic","components"]).iloc[0].components)
    table["selected"] = table.components.eq(best_k) & table.valid
    return table, fitted[best_k], scaler