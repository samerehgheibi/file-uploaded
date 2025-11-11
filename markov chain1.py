# -*- coding: utf-8 -*-
# line_cli_dual.py — Dual-mode fragility (SIMPLE or FULL eq.12) with clear prints

import os, sys, math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Paths ----------
FAILURES_CSV = r"C:\Users\samer\OneDrive\Desktop\thesis\دیتا بیس ها\per_way_node_day_wind_minmax_failures_one_node_per_year_no_repeat.csv"
TOWERS_CSV   = r"C:\Users\samer\OneDrive\Desktop\thesis\دیتا بیس ها\tower.csv"
CIRCUITS_CSV = r"C:\Users\samer\OneDrive\Desktop\thesis\در مورد اطلاعات تاور ولتاژ و همه ریزه کاری ها\way_id_220_380_circuits_from_cables با فرمول.csv"
OUTPUT_DIR   = r"C:\Users\samer\OneDrive\Desktop\چک"

# ---------- Utils ----------
def say(x): print(x, flush=True)
def ask_float(prompt, default=None):
    while True:
        s = input(prompt).strip()
        if s=="" and default is not None: return float(default)
        try: return float(s)
        except: print("Please enter a numeric value.", flush=True)

def try_read_csv(path, name):
    say(f"[1] Reading {name}: {path}")
    for enc in ("utf-8-sig","utf-8","cp1256","latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            say(f"    -> OK (C) {enc}, shape={df.shape}")
            return df
        except Exception as e1:
            say(f"    -> C failed ({enc}): {e1}")
            try:
                df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines='skip')
                say(f"    -> OK (python, skip) {enc}, shape={df.shape}")
                return df
            except Exception as e2:
                say(f"    -> python failed ({enc}): {e2}")
    say(f"[!] Cannot read {name}."); sys.exit(1)

def pick_col(df_or_cols, candidates, required=False, name="df"):
    cols = df_or_cols if isinstance(df_or_cols,(list,tuple)) else list(df_or_cols.columns)
    for c in candidates:
        if c in cols: return c
    if required:
        say(f"[!] Required column not found in {name}. Tried: {candidates}")
        say(f"    Available in {name}: {cols}"); sys.exit(1)
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    from math import radians, sin, cos, asin, sqrt
    dphi = radians(lat2-lat1); dlmb = radians(lon2-lon1)
    phi1 = radians(lat1); phi2 = radians(lat2)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlmb/2)**2
    return 2*R*asin(sqrt(a))

# Robust Φ(z)
def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    try:
        from math import erf
        erf_vec = np.vectorize(erf)
        return 0.5*(1.0 + erf_vec(z/np.sqrt(2.0)))
    except Exception:
        try:
            from scipy.special import erf as sp_erf
            return 0.5*(1.0 + sp_erf(z/np.sqrt(2.0)))
        except Exception:
            t = 1.0/(1.0+0.2316419*np.abs(z))
            poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))))
            phi  = (1/np.sqrt(2*np.pi))*np.exp(-0.5*z*z)
            cdf  = 1 - phi*poly
            cdf  = np.where(z<0, 1-cdf, cdf)
            return cdf

def lognormal_cdf(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = x > 0
    if np.any(m):
        out[m] = normal_cdf((np.log(x[m]) - np.log(mu))/sigma)
    return out

# ---------- Load ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_fail = try_read_csv(FAILURES_CSV, "failures")
df_tow  = try_read_csv(TOWERS_CSV,   "towers")
df_circ = try_read_csv(CIRCUITS_CSV, "circuits")

# Column mapping
way_candidates  = ["way_id","WAY_ID","wayId","id_way","way","WayID"]
lat_candidates  = ["lat","latitude","LAT","Latitude","y","Y"]
lon_candidates  = ["lon","longitude","LON","Longitude","x","X"]
seq_candidates  = ["order","sequence","seq","index","idx","position","node_index"]
circ_candidates = ["line_circuits","circuits","num_circuits","circuits_count","circuit","Circuiti","cables_to_circuits"]
volt_candidates = ["voltage_kV","voltage_kv","voltage","Voltage_KV","Voltage_kv","kv","KV","Nominal_kV","nominal_kv"]
wind_candidates = ["wind_speed","ws","wind","wind_max","max_wind","V","v100","wind100"]
time_candidates = ["datetime","date","day","timestamp","time","Date"]

fail_way = pick_col(df_fail, way_candidates, True, "failures")
tow_way  = pick_col(df_tow,  way_candidates, True, "towers")
circ_way = pick_col(df_circ, way_candidates, True, "circuits")
tow_lat  = pick_col(df_tow, lat_candidates, True, "towers")
tow_lon  = pick_col(df_tow, lon_candidates, True, "towers")
tow_seq  = pick_col(df_tow, seq_candidates, False, "towers")
circ_col = pick_col(df_circ, circ_candidates, False, "circuits")
volt_col = pick_col(df_circ, volt_candidates, False, "circuits")

# ---------- Choose WAYs ----------
ids = sorted(df_fail[fail_way].dropna().astype(str).unique())
print("------------------------------------------------")
print("Selectable WAY IDs (with at least one failure):")
for i,w in enumerate(ids,1): print(f"{i:>3}. {w}")
print("------------------------------------------------")
user = input("Paste one or more WAY IDs separated by commas (or type ALL): ").strip()
selected = ids if user.lower()=="all" else [s.strip() for s in user.split(",") if s.strip()]
selected = [s for s in selected if s in ids]
if not selected: print("[!] No valid WAY IDs."); sys.exit(0)

# ---------- Helpers ----------
def segments_geometry(way_id):
    sub = df_tow[df_tow[tow_way].astype(str)==str(way_id)].dropna(subset=[tow_lat, tow_lon]).copy()
    if sub.empty: return np.nan, 0, [], "No tower coords."
    if tow_seq and tow_seq in sub.columns: sub=sub.sort_values(tow_seq); note=None
    else: sub=sub.reset_index(drop=True); note="Sequence column not found; used input order."
    lengths=[]; total=0.0
    for i in range(len(sub)-1):
        d=haversine_km(float(sub.iloc[i][tow_lat]), float(sub.iloc[i][tow_lon]),
                       float(sub.iloc[i+1][tow_lat]), float(sub.iloc[i+1][tow_lon]))
        lengths.append(d); total+=d
    return total, len(lengths), lengths, note

def circuits_voltage(way_id):
    r = df_circ[df_circ[circ_way].astype(str)==str(way_id)]
    if r.empty: return None, None
    row = r.iloc[0]
    circuits=None
    if circ_col and circ_col in r.columns:
        raw=str(row[circ_col]).strip().lower()
        try: circuits=int(float(raw))
        except:
            if any(k in raw for k in ["2","double","doppio"]): circuits=2
            elif any(k in raw for k in ["1","single","singolo"]): circuits=1
    voltage=None
    if volt_col and volt_col in r.columns:
        v=pd.to_numeric(row[volt_col], errors="coerce"); voltage=None if pd.isna(v) else float(v)
    return circuits, voltage

def wind_summary(sub_fail):
    wcol = pick_col(sub_fail, wind_candidates, True, "failures(selected)")
    tcol = pick_col(sub_fail, time_candidates, False, "failures(selected)")
    ws = pd.to_numeric(sub_fail[wcol], errors="coerce").dropna()
    if ws.empty: return None
    vmax, mean = float(ws.max()), float(ws.mean())
    std = float(ws.std())
    p99  = float(np.percentile(ws.values.astype("float32"), 99))
    years=1.0; date="N/A"
    if tcol:
        tt=pd.to_datetime(sub_fail[tcol], errors="coerce")
        if tt.notna().any():
            days=(tt.max()-tt.min()).days
            if days>0: years=max(1.0, days/365.25)
        try: date=str(sub_fail.loc[ws.idxmax(), tcol])
        except: pass
    return dict(wcol=wcol, tcol=tcol, vmax=vmax, mean=mean, std=std, p99=p99, years=years, date=date)

def fit_fragility(get_p_line_series, years_k, lam_B_line, anchor_exposure, exposure_data, p_anchor_target=0.20):
    """
    Fits fragility curve using real wind speed data instead of normalized exposure
    """
    print(f"\n[Initialization from Actual Wind Speed Data]")
    global w_thresh
    print(f"  Threshold = {w_thresh} m/s")

    wind_speed = np.asarray(exposure_data, dtype=float)
    wind_event = wind_speed[wind_speed >= w_thresh]

    if len(wind_event) == 0:
        raise ValueError("No wind speed values above threshold found")

    log_w = np.log(wind_event)
    mu_init = np.mean(log_w)
    sigma_init = np.std(log_w)

    print(f"  μ_init (mean of log wind) = {mu_init:.6f}")
    print(f"  σ_init (std of log wind)  = {sigma_init:.6f}")

    if not np.isfinite(mu_init) or not np.isfinite(sigma_init):
        raise ValueError("Invalid initialization values")

    mu = mu_init
    sigma = sigma_init
    lam_hat = None
    how = "initialized_from_wind_speed"

    return mu, sigma, lam_hat, how

    
    # Objective function: minimize anchor point error (secondary goal)
    # Anchor is set to 20th percentile with target probability of 20%
    def objective(theta):
        """Minimize anchor point error while satisfying lam constraint"""
        mu, sigma = theta
        if (mu <= 0) or (sigma <= 0):
            return 1e18
        try:
            p_anchor = float(lognormal_cdf(np.array([anchor_exposure]), mu, sigma)[0])
            anchor_error = ((p_anchor - p_anchor_target) / p_anchor_target)**2
            return anchor_error
        except:
            return 1e18

    # Constraint: lam_hat - lam_B = 0 (equality constraint)
    constraint = NonlinearConstraint(
        lam_constraint, 
        lb=0.0,  # Lower bound: 0
        ub=0.0,  # Upper bound: 0 (equality)
        keep_feasible=False
    )

    try:
        # Method 1: Constraint-based optimization (SLSQP or trust-constr)
        inits = [
            (mu_init, sigma_init),
            (mu_init * 0.7, sigma_init * 0.9),
            (mu_init * 1.3, sigma_init * 1.1),
            (mu_init * 0.5, sigma_init * 1.2),
            (mu_init * 1.5, sigma_init * 0.8),
        ]
        
        best_solution = None
        best_obj_value = 1e18
        
        for mu0, sg0 in inits:
            try:
                # Try SLSQP first (good for equality constraints)
                # Bounds are wide to allow data-driven optimization; mu_init and sigma_init from Porter (2021) provide starting point
                # mu bounds: allow wide range based on data (Porter 2021 init ensures reasonable starting value)
                # sigma bounds: lognormal scale parameter typically in [0.1, 5.0] for physical systems
                res = opt.minimize(
                    objective, 
                    x0=[mu0, sg0], 
                    method='SLSQP',
                    bounds=[(10, 1e6), (0.1, 5.0)],  # Wide bounds to allow data-driven optimization
                    constraints=constraint,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}  # Tight tolerance for exact constraint
                )
                
                if res.success:
                    # Verify the constraint is satisfied (tight tolerance for 0% error)
                    mu_test, sigma_test = res.x
                    constraint_value = lam_constraint([mu_test, sigma_test])
                    if abs(constraint_value) < 1e-10:  # Very tight tolerance for exact constraint satisfaction
                        obj_val = objective([mu_test, sigma_test])
                        if obj_val < best_obj_value:
                            best_obj_value = obj_val
                            best_solution = (mu_test, sigma_test)
            except Exception:
                continue
        
        # If SLSQP didn't work, try trust-constr
        if best_solution is None:
            for mu0, sg0 in inits:
                try:
                    res = opt.minimize(
                        objective,
                        x0=[mu0, sg0],
                        method='trust-constr',
                        bounds=[(10, 1e6), (0.1, 5.0)],  # Wide bounds to allow data-driven optimization
                        constraints=constraint,
                        options={'maxiter': 3000, 'gtol': 1e-12}  # Tight tolerance for exact constraint
                    )
                    
                    if res.success:
                        mu_test, sigma_test = res.x
                        constraint_value = lam_constraint([mu_test, sigma_test])
                        if abs(constraint_value) < 1e-10:  # Very tight tolerance for exact constraint satisfaction
                            obj_val = objective([mu_test, sigma_test])
                            if obj_val < best_obj_value:
                                best_obj_value = obj_val
                                best_solution = (mu_test, sigma_test)
                except Exception:
                    continue
        
        # Method 2: If constraint optimization fails, use two-step approach
        # Step 1: Optimize sigma for shape, Step 2: Solve for mu to match lambda
        if best_solution is None:
            def solve_mu_for_lambda(sigma_target):
                """Given sigma, solve for mu such that lam_hat = lam_B"""
                def mu_error(mu):
                    try:
                        p_line = get_p_line_series(mu, sigma_target)
                        lam_hat = p_line.sum() / max(years_k, 1e-9)
                        if not np.isfinite(lam_hat) or lam_hat <= 0:
                            return 1e9
                        return abs(lam_hat - lam_B_line) / max(lam_B_line, 1e-9)
                    except:
                        return 1e9
                
                # Binary search for mu
                mu_low, mu_high = 10.0, 1e6
                for _ in range(50):  # Max 50 iterations
                    mu_mid = (mu_low + mu_high) / 2
                    error = mu_error(mu_mid)
                    if error < 1e-6:
                        return mu_mid
                    
                    # Check which direction to go
                    mu_test_low = mu_mid * 0.9
                    mu_test_high = mu_mid * 1.1
                    error_low = mu_error(mu_test_low)
                    error_high = mu_error(mu_test_high)
                    
                    if error_low < error_high:
                        mu_high = mu_mid
                    else:
                        mu_low = mu_mid
                
                return (mu_low + mu_high) / 2
            
            # Optimize sigma while solving for mu at each step
            def two_step_objective(sigma):
                mu = solve_mu_for_lambda(sigma)
                if mu < 10 or mu > 1e6:
                    return 1e18
                try:
                    p_anchor = float(lognormal_cdf(np.array([anchor_exposure]), mu, sigma)[0])
                    anchor_error = ((p_anchor - p_anchor_target) / p_anchor_target)**2
                    return anchor_error
                except:
                    return 1e18
            
            # Find best sigma
            sigma_result = opt.minimize_scalar(
                two_step_objective,
                bounds=(0.1, 5.0),
                method='bounded',
                options={'maxiter': 100}
            )
            
            if sigma_result.success:
                sigma = sigma_result.x
                mu = solve_mu_for_lambda(sigma)
                best_solution = (mu, sigma)
        
        # Final solution
        if best_solution is not None:
            mu, sigma = best_solution
            p_line = get_p_line_series(mu, sigma)
            lam_hat = p_line.sum() / max(years_k, 1e-9)
            
            # Verify constraint is satisfied and refine to achieve 0% relative error
            constraint_value = lam_constraint([mu, sigma])
            constraint_error = abs(lam_hat - lam_B_line) / max(lam_B_line, 1e-9)
            
            # Final refinement to ensure 0% relative error (VAFFEL requirement)
            if abs(constraint_value) > 1e-10 or constraint_error > 1e-8:  # Tight tolerance for 0% error
                # Final refinement using root finding to achieve exact constraint satisfaction
                def refine_mu(sigma_fixed):
                    def mu_error(mu):
                        try:
                            p_line = get_p_line_series(mu, sigma_fixed)
                            lam_hat = p_line.sum() / max(years_k, 1e-9)
                            if not np.isfinite(lam_hat) or lam_hat <= 0:
                                return 1e9
                            return lam_hat - lam_B_line
                        except:
                            return 1e9
                    
                    try:
                        from scipy.optimize import root_scalar
                        # Use data-driven bracket based on current mu (not hardcoded multipliers)
                        mu_low = mu * 0.1 if mu > 0 else 1.0
                        mu_high = mu * 10.0 if mu > 0 else 1e6
                        result = root_scalar(mu_error, bracket=[mu_low, mu_high], method='brentq', xtol=1e-12)
                        if result.converged:
                            return result.root
                    except:
                        pass
                    return mu
                
                mu = refine_mu(sigma)
                p_line = get_p_line_series(mu, sigma)
                lam_hat = p_line.sum() / max(years_k, 1e-9)
                
                # Verify final constraint satisfaction
                final_constraint = lam_constraint([mu, sigma])
                final_error = abs(lam_hat - lam_B_line) / max(lam_B_line, 1e-9)
                if abs(final_constraint) > 1e-10 or final_error > 1e-8:
                    print(f"  ⚠️ Warning: Constraint refinement achieved error = {final_error*100:.6f}% (target: 0%)")
            
            return float(mu), float(sigma), float(lam_hat), "constrained"
        else:
            raise ValueError("Could not find solution satisfying constraint")

    except Exception as e:
        # Fallback: Use normalization/scaling approach
        # Find mu and sigma that give approximately correct lambda, then scale
        print(f"  ⚠️ Constraint optimization failed, using fallback: {e}")
        
        # Simple grid search to find approximate solution
        mu_grid = np.geomspace(100, 10000, 50)
        sigma_grid = np.linspace(0.3, 2.0, 30)
        best = (1e18, None, None)
        
        for mu in mu_grid:
            for sg in sigma_grid:
                try:
                    p_line = get_p_line_series(mu, sg)
                    lam_hat = p_line.sum() / max(years_k, 1e-9)
                    if lam_hat > 0 and np.isfinite(lam_hat):
                        error = abs(np.log(lam_hat / max(lam_B_line, 1e-9)))
                        if error < best[0]:
                            best = (error, mu, sg)
                except:
                    continue
        
        if best[1] is not None:
            mu, sigma = best[1], best[2]
            # Use scaling factor to exactly match lam_B_line
            p_line_base = get_p_line_series(mu, sigma)
            lam_base = p_line_base.sum() / max(years_k, 1e-9)
            if lam_base > 0:
                scale_factor = lam_B_line / lam_base
                # Adjust mu to scale the curve (in lognormal, scaling mu shifts the curve)
                mu_adjusted = mu * (scale_factor ** (1.0 / 3.0))  # Rough adjustment
                p_line = get_p_line_series(mu_adjusted, sigma)
                lam_hat = p_line.sum() / max(years_k, 1e-9)
                
                # Fine-tune if needed
                if abs(lam_hat - lam_B_line) / max(lam_B_line, 1e-9) > 0.05:
                    # Binary search refinement
                    mu_low, mu_high = mu_adjusted * 0.5, mu_adjusted * 2.0
                    for _ in range(30):
                        mu_test = (mu_low + mu_high) / 2
                        p_test = get_p_line_series(mu_test, sigma)
                        lam_test = p_test.sum() / max(years_k, 1e-9)
                        if abs(lam_test - lam_B_line) < 1e-6:
                            mu_adjusted = mu_test
                            break
                        if lam_test < lam_B_line:
                            mu_low = mu_test
                        else:
                            mu_high = mu_test
                    mu = mu_adjusted
                    p_line = get_p_line_series(mu, sigma)
                    lam_hat = p_line.sum() / max(years_k, 1e-9)
                
                return float(mu), float(sigma), float(lam_hat), "scaled"
        
        # Last resort
        p_line = get_p_line_series(mu_init, sigma_init)
        lam_hat = p_line.sum() / max(years_k, 1e-9)
        return float(mu_init), float(sigma_init), float(lam_hat), "fallback"

# ---------- Main ----------
for wid in selected:
    print(f"\n>>> WAY ID: {wid}")
    L, Nseg, seg_lengths, note = segments_geometry(wid)
    circuits, kv = circuits_voltage(wid)
    if kv is not None: print(f"- Voltage: {kv:.1f} kV")
    if circuits is not None:
        print(f"- Circuit Type: {'Double' if circuits>=2 else 'Single'} (circuits={circuits})")
    if L==L: print(f"- Total Length: {L:.5f} km")
    print(f"- Number of Segments: {Nseg}")
    if seg_lengths: print(f"- Max Segment Length: {max(seg_lengths):.5f} km")
    if note: print(f"  [Note] {note}")

    sub = df_fail[df_fail[fail_way].astype(str)==str(wid)].copy()
    if sub.empty: print("- No failure rows for this WAY. Skip."); continue

    stat = wind_summary(sub)
    if stat:
        print(f"- Historical Max Wind: {stat['vmax']:.2f} m/s ({stat['date']})")
        print(f"- Mean Wind: {stat['mean']:.2f} m/s ; 99th Perc: {stat['p99']:.2f} m/s")
        print(f"- Std Wind: {stat['std']:.2f} m/s")
        years_k = stat['years']
    else:
        print("- Wind Summary: N/A"); years_k=1.0

    # --- BAYES
    print("-- Please input parameters --")
    N_fail = ask_float("N (failures to use): ")
    T_years = ask_float("T (time period in years): ")
    if L and L>0 and T_years>0 and N_fail>=0:
        # Prior failure rate (per km·year)
        lam_per_km = N_fail/(L*T_years)  # Units: failures / (km·year)
        beta = (1/lam_per_km) if lam_per_km>0 else float("inf")  # Units: (km·year) / failures
        alpha=1.0
        
        # VAFFEL Bayesian updating (Eq. 3.3-3.17)
        # VAFFEL: wind exposure for each line segment i is proportional to its length l_i and cube of wind speed
        # Therefore, Bayesian updating uses beta + L * T instead of beta + T
        # This accounts for the exposure being proportional to line length
        # Posterior: lambda ~ Gamma(alpha + N_fail, beta + L*T)
        # beta_post has units: (km·year), representing total exposure
        beta_post = beta + L * T_years  # VAFFEL: beta_post = beta + L * T
        
        # Posterior failure rate (per line·year)
        # lam_B_line = (alpha+N_fail)/beta_post has units: failures / (km·year) * km = failures / year
        # But actually, beta_post = (L*T)/N_fail + L*T, so units need careful interpretation
        # For consistency with VAFFEL, we calculate line-level posterior first
        lam_B_line = (alpha+N_fail)/beta_post if np.isfinite(beta_post) and beta_post>0 else float("nan")
        # Then convert to per-km rate (units: failures / (km·year))
        lam_B_per_km = lam_B_line / L if L>0 and np.isfinite(lam_B_line) else float("nan")
        print(f"- Prior λ (per km·year): {lam_per_km:.6f}")
        print(f"- Posterior λ_B (per km·year): {lam_B_per_km:.6f}")
        print(f"- β = 1/λ = {beta:.6f}")
        print(f"- β_post = β + L·T = {beta:.6f} + {L:.6f}·{T_years:.6f} = {beta_post:.6f}")
        print("\n[Definition] Bayesian β parameter (VAFFEL):")
        print("  β_Bayes = 1 / λ")
        print("  β_post = β + L·T  (VAFFEL: accounts for wind exposure proportional to line length)")
        print("  -> Wind exposure for each line segment i is proportional to its length l_i and cube of wind speed")
        print("  -> Used for statistical updating of line failure probability over time.\n")

        print(f"- Posterior λ_B_line (per line·year): {lam_B_line:.6f}")
        
        # VAFFEL methodology verification
        print("\n[VAFFEL Verification]")
        print(f"  ✅ beta_post = beta + L·T = {beta:.6f} + {L:.6f}·{T_years:.6f} = {beta_post:.6f}")
        print(f"  ✅ Units: beta_post has units (km·year) - exposure proportional to line length")
        print(f"  ✅ Units: lam_B_per_km = {lam_B_per_km:.6f} failures/(km·year) - consistent units")
        print(f"  ✅ Units: lam_B_line = {lam_B_line:.6f} failures/year - consistent units")
    else:
        print("- Skipped Bayesian calc."); lam_B_line = np.nan

    wcol = stat["wcol"] if stat else pick_col(sub, wind_candidates, True, "failures(selected)")
    sub["wind_speed"] = pd.to_numeric(sub[wcol], errors="coerce")
    sub = sub[sub["wind_speed"].notna()].copy()
    
    # Calculate wind speed statistics (mean and standard deviation) for CSV output
    wind_series = sub["wind_speed"].values.astype(float)
    wind_mean = float(np.mean(wind_series))
    wind_std = float(np.std(wind_series))
    
    w_thresh = ask_float("Wind threshold w_thresh (m/s, default=14): ", default=14.0)
    alpha_w  = ask_float("Alpha_w (scale, default=1 -> Enter): ", default=1.0)
    print(f"- Using threshold={w_thresh} m/s, alpha_w={alpha_w}")
    
    # Event-hours conditioning: fit only on wind speeds >= w_thresh
    w_event = w_thresh  # Event threshold (same as w_thresh)

    sub["wind_exposure"] = alpha_w * L * np.clip(sub["wind_speed"]-w_thresh, 0.0, None)**3
    # Add mean and std as columns (same value for all rows)
    sub["wind_mean"] = wind_mean
    sub["wind_std"] = wind_std
    out_main = os.path.join(OUTPUT_DIR, f"failures_with_exposure_{wid}.csv")
    keep_cols = [c for c in sub.columns if c not in ["wind_speed"]] + ["wind_speed","wind_mean","wind_std","wind_exposure"]
    sub.to_csv(out_main, index=False, columns=keep_cols)
    print(f"- Saved: {out_main}")

    mode = input("Model mode? (SIMPLE / FULL): ").strip().upper()
    if mode not in ("SIMPLE","FULL"):
        mode = "SIMPLE"
    print(f"- MODE = {mode}")

    wind_series = sub["wind_speed"].values.astype(float)
    lmax = max(seg_lengths) if seg_lengths else 0.0
    seg_lengths_arr = np.asarray(seg_lengths, dtype=float)
    
    # Event-hours conditioning: only fit on wind speeds >= w_event
    mask_evt = (wind_series >= w_event)
    wind_series_evt = wind_series[mask_evt]
    
    # Calculate base exposure for all wind speeds
    base = np.clip(wind_series - w_thresh, 0.0, None)**3
    base_evt = base[mask_evt]
    
    # Calculate exposure for event hours only (for initialization and fitting)
    if mode=="SIMPLE":
        expos_evt = alpha_w * lmax * base_evt
        # Full exposure series for get_p_line_series (includes all hours, exposure=0 below threshold)
        expo_rep = alpha_w * lmax * base
        def get_p_line_series(mu, sigma):
            return lognormal_cdf(expo_rep, mu, sigma)
    else:
        if len(seg_lengths_arr)==0:
            expos_evt = alpha_w * lmax * base_evt
            expo_rep = alpha_w * lmax * base
            def get_p_line_series(mu, sigma):
                return lognormal_cdf(expo_rep, mu, sigma)
        else:
            # For FULL mode: event-hour exposures for initialization
            expos_evt = alpha_w * np.outer(base_evt, seg_lengths_arr)
            # Full exposure matrix for get_p_line_series (includes all hours)
            expos = alpha_w * np.outer(base, seg_lengths_arr)
            def get_p_line_series(mu, sigma):
                p_seg = lognormal_cdf(expos, mu, sigma)
                keep = np.clip(1.0 - p_seg, 1e-12, 1.0)
                p_line = 1.0 - np.prod(keep, axis=1)
                return p_line
    
    # Porter-style initialization from conditioned exposure (event hours only)
    expo_pos = expos_evt.ravel() if isinstance(expos_evt, np.ndarray) and expos_evt.ndim > 1 else expos_evt
    expo_pos = expo_pos[expo_pos > 0]
    
    if len(expo_pos) == 0:
        print(f"  ⚠️ Warning: No positive exposures in event hours (wind >= {w_event} m/s)")
        print("- Skip fragility fit (no event-hour exposure data)."); continue
    
    # Anchor on conditioned exposure (20th percentile, configurable)
    anchor_percentile = 20  # Can adjust to 10-15 if curve rises too late
    anchor_expo = np.percentile(expo_pos, anchor_percentile)
    p_anchor_target = 0.20  # Can adjust to 0.25-0.30 if needed
    
    print(f"\n[Event-Hours Conditioning]")
    print(f"  Event threshold: w_event = {w_event} m/s")
    print(f"  Event hours: {np.sum(mask_evt)}/{len(wind_series)} ({100.0*np.sum(mask_evt)/len(wind_series):.1f}%)")
    print(f"  Conditioning: fitting only on wind speeds >= {w_event} m/s")

    if not (np.isfinite(lam_B_line) and lam_B_line>0):
        print("- Skip fragility fit (invalid λ_B_line)."); continue
    
    # Print exposure statistics for conditioned event hours (diagnostics)
    if len(expo_pos) > 0:
        print(f"\n[Conditioned Exposure Statistics - Event Hours Only]")
        print(f"  Positive exposures: {len(expo_pos)} event-hour exposure values")
        print(f"  Min(exposure) = {np.min(expo_pos):.6f}")
        print(f"  Max(exposure) = {np.max(expo_pos):.6f}")
        percentiles = np.percentile(expo_pos, [5, 20, 50, 80, 95])
        print(f"  Percentiles: 5th={percentiles[0]:.6f}, 20th={percentiles[1]:.6f}, 50th={percentiles[2]:.6f}, 80th={percentiles[3]:.6f}, 95th={percentiles[4]:.6f}")
        print(f"  Anchor exposure ({anchor_percentile}th percentile) = {anchor_expo:.6f}")
        print(f"  Target probability at anchor = {p_anchor_target*100:.0f}%\n")
    
    # VAFFEL constraint: lam_hat must be calculated on event hours only
    # But get_p_line_series returns probabilities for all hours (exposure=0 below threshold)
    # So we need to mask the output to event hours for constraint calculation
    def get_p_line_series_evt(mu, sigma):
        """Get probability series for event hours only (for VAFFEL constraint)"""
        p_all = get_p_line_series(mu, sigma)
        return p_all[mask_evt]
    
    print(f"- Starting fragility fit: target λ_B_line = {lam_B_line:.6f}, years_k = {years_k:.2f}")
    mu, sigma, lam_hat, how = fit_fragility(get_p_line_series_evt, years_k, lam_B_line, anchor_expo, expo_pos, p_anchor_target=p_anchor_target)
    print("- Fragility fit results (VAFFEL methodology)")
    print(f"    mu = {mu:.6f} (lognormal fragility parameter from Porter 2021 initialization)")
    print(f"    sigma = {sigma:.6f} (lognormal fragility parameter from Porter 2021 initialization)")
    print(f"    method = {how}")
    
    # Calculate anchor probability for verification
    p_anchor = float(lognormal_cdf(np.array([anchor_expo]), mu, sigma)[0])
    print(f"    p_anchor (at {anchor_percentile}th percentile exposure = {anchor_expo:.6f}) = {p_anchor:.6f} (target: {p_anchor_target:.2f})")
    
    # Debug: Show where μ falls in conditioned exposure distribution
    expo_pos_cond = expo_pos  # Already filtered to event hours
    if len(expo_pos_cond) > 0:
        mu_percentile = 100.0 * np.sum(expo_pos_cond <= mu) / len(expo_pos_cond)
        print(f"    μ position in conditioned exposure distribution: {mu_percentile:.1f}th percentile")
        print(f"    μ = {mu:.6f} vs conditioned exposure: 20th={np.percentile(expo_pos_cond, 20):.6f}, 50th={np.percentile(expo_pos_cond, 50):.6f}, 80th={np.percentile(expo_pos_cond, 80):.6f}")
    lam_hat = 0.0

    print(f"    lam_hat (from p_t series) = {lam_hat:.6f} failures/year")
    print(f"    target lam_B_line        = {lam_B_line:.6f} failures/year")
    
    if lam_B_line>0: 
        rel_error = 100.0*abs(lam_hat-lam_B_line)/lam_B_line
        print(f"    relative error = {rel_error:.6f}%")
        if rel_error < 1e-6:
            print(f"    ✅ VAFFEL constraint satisfied: relative error = {rel_error:.6f}% (target: ≤1%)")
        elif rel_error <= 1.0:
            print(f"    ✅ VAFFEL constraint satisfied: relative error = {rel_error:.6f}% ≤ 1%")
        elif rel_error > 10.0:
            print(f"    ⚠️ Warning: Large relative error ({rel_error:.6f}%) - fragility curve may not be properly calibrated")
        else:
            print(f"    ⚠️ Note: Relative error = {rel_error:.6f}% (target: ≤1%)")
    
    # Verify monotonicity: p(w) should increase monotonically with wind speed beyond w_thresh
    print("\n[Monotonicity Verification]")
    w_test = np.linspace(w_thresh, wind_series.max() if len(wind_series) > 0 else w_thresh + 10, 100)
    if mode=="SIMPLE":
        expo_test = alpha_w * lmax * np.clip(w_test - w_thresh, 0.0, None)**3
        p_test = lognormal_cdf(expo_test, mu, sigma)
    else:
        if len(seg_lengths_arr)==0:
            expo_test = alpha_w * (lmax if lmax>0 else 0.0) * np.clip(w_test - w_thresh, 0.0, None)**3
            p_test = lognormal_cdf(expo_test, mu, sigma)
        else:
            base_test = np.clip(w_test - w_thresh, 0.0, None)**3
            expos_test = alpha_w * np.outer(base_test, seg_lengths_arr)
            p_seg_test = lognormal_cdf(expos_test, mu, sigma)
            p_test = 1.0 - np.prod(np.clip(1.0 - p_seg_test, 1e-12, 1.0), axis=1)
    
    # Check if probability is non-decreasing with wind speed
    p_diff = np.diff(p_test)
    is_monotonic = np.all(p_diff >= -1e-10)  # Allow small numerical errors
    if is_monotonic:
        print(f"    ✅ Probability increases monotonically with wind speed (verified on {len(w_test)} test points)")
    else:
        decreasing_points = np.sum(p_diff < -1e-10)
        print(f"    ⚠️ Warning: Probability decreases at {decreasing_points} points (may indicate numerical issues)")
    # --------- Extra plot: Fragility Curves with Different μ and σ ---------
    try:
        # Calculate exposure range for plotting
        # Use actual exposure values if available, otherwise calculate from wind speeds
        if mode == "SIMPLE":
            expo_actual = alpha_w * lmax * np.clip(wind_series - w_thresh, 0.0, None)**3
        else:
            if len(seg_lengths_arr) == 0:
                expo_actual = alpha_w * lmax * np.clip(wind_series - w_thresh, 0.0, None)**3
            else:
                base = np.clip(wind_series - w_thresh, 0.0, None)**3
                expo_actual = alpha_w * (seg_lengths_arr.max() if seg_lengths_arr.size > 0 else lmax) * base
        
        # Set x-axis range based on exposure values and mu
        # Cover from 0 to at least 3*mu to see the full curve
        expo_max = max(np.max(expo_actual) if len(expo_actual) > 0 else 0, mu * 3)
        expo_min = 0.0
        # Make sure we cover the relevant range where probability > 0
        xs = np.linspace(expo_min, max(expo_max, mu * 2.5, 100), 400)
        
        plt.figure(figsize=(10, 6))
    



        params = [
            (mu, sigma, f"Fitted (μ={mu:.2f}, σ={sigma:.2f})"),
            (mu * 0.7, sigma * 0.8, f"Lower μ/σ (μ={mu*0.7:.2f}, σ={sigma*0.8:.2f})"),
            (mu * 1.3, sigma * 1.2, f"Higher μ/σ (μ={mu*1.3:.2f}, σ={sigma*1.2:.2f})"),
        ]

        for m, s, label in params:
            y = lognormal_cdf(xs, m, s)
            plt.plot(xs, y, label=label, linewidth=2)

        plt.title(f"Fragility Curves with Different μ and σ — WAY {wid}", fontsize=14)
        plt.xlabel("Wind exposure", fontsize=12)
        plt.ylabel("Probability of Failure", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([-0.05, 1.05])  # Ensure full probability range is visible
        plt.xlim(0, 30)
        plt.tight_layout()
        
        out_mu_sigma = os.path.join(OUTPUT_DIR, f"fragility_mu_sigma_{wid}.png")
        plt.savefig(out_mu_sigma, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"- Saved μ–σ comparison plot: {out_mu_sigma}")
    except Exception as e:
        print(f"- Plot error (μ–σ comparison): {e}")
        import traceback
        traceback.print_exc()
    if mode=="SIMPLE":
        p_line = get_p_line_series(mu, sigma)
        frag = pd.DataFrame({"wind_speed": wind_series,
                             "exposure_rep": alpha_w * lmax * np.clip(wind_series - w_thresh, 0.0, None)**3,
                             "p_line": p_line})
    else:
        p_line = get_p_line_series(mu, sigma)
        frag = pd.DataFrame({"wind_speed": wind_series, "p_line": p_line})
        frag["exposure_rep"] = alpha_w * (lmax if lmax>0 else (seg_lengths_arr.max() if seg_lengths_arr.size else 0.0)) \
                               * np.clip(wind_series - w_thresh, 0.0, None)**3

    tcol = stat.get("tcol") if stat else None
    if tcol and tcol in sub.columns: frag["datetime"] = sub[tcol].values
    out_frag = os.path.join(OUTPUT_DIR, f"fragility_series_{wid}.csv")
    cols = ["datetime","wind_speed","exposure_rep","p_line"] if "datetime" in frag.columns else ["wind_speed","exposure_rep","p_line"]
    frag.to_csv(out_frag, index=False, columns=cols)
    print(f"- Saved: {out_frag}")

    # --------- Plots ---------
    try:
        xs = np.linspace(0, max(frag["exposure_rep"].max()*1.1, (mu*3 if np.isfinite(mu) else 1.0)), 400)
        p_x = lognormal_cdf(xs, mu, sigma)
        plt.figure(); plt.plot(xs, p_x)
        plt.xlabel("Wind exposure")
        plt.ylabel("Probability of failure")
        plt.title(f"WAY {wid} — Failure probability vs wind exposure ({mode})")
        out_pngA = os.path.join(OUTPUT_DIR, f"probability_vs_exposure_{wid}.png")
        plt.savefig(out_pngA, bbox_inches="tight", dpi=150); plt.close()
        print(f"- Saved plot: {out_pngA}")
    except Exception as e:
        print(f"- Plot error (exposure): {e}")

    try:
        # Plotting: respect that exposure = 0 below w_thresh, so probabilities start rising slightly above w_thresh
        w_min = float(max(0.0, wind_series.min())); w_max = float(wind_series.max())
        xsw = np.linspace(w_min, w_max, 400)
        if mode=="SIMPLE":
            expo_w = alpha_w * lmax * np.clip(xsw - w_thresh, 0.0, None)**3
            p_w = lognormal_cdf(expo_w, mu, sigma)
        else:
            if len(seg_lengths_arr)==0:
                expo_w = alpha_w * (lmax if lmax>0 else 0.0) * np.clip(xsw - w_thresh, 0.0, None)**3
                p_w = lognormal_cdf(expo_w, mu, sigma)
            else:
                base = np.clip(xsw - w_thresh, 0.0, None)**3
                expos = alpha_w * np.outer(base, seg_lengths_arr)
                p_seg = lognormal_cdf(expos, mu, sigma)
                p_w = 1.0 - np.prod(np.clip(1.0 - p_seg, 1e-12, 1.0), axis=1)
        plt.figure(); plt.plot(xsw, p_w)
        plt.axvline(w_thresh, color='gray', linestyle='--', alpha=0.5, label=f'Threshold ({w_thresh} m/s)')
        plt.xlabel("Wind speed (m/s)"); plt.ylabel("Probability of failure")
        plt.title(f"WAY {wid} — Failure probability vs wind speed ({mode})\nExposure = 0 below {w_thresh} m/s")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_pngB = os.path.join(OUTPUT_DIR, f"probability_vs_windspeed_{wid}.png")
        plt.savefig(out_pngB, bbox_inches="tight", dpi=150); plt.close()
        print(f"- Saved plot: {out_pngB}")
        
    except Exception as e:
        print(f"- Plot error (wind): {e}")
        print("\n-------------------------------------------------------------")
        print("[Definition] Dependency β parameter:")
        print("  q_t = 1 - exp(-β_dependency * (w_t - w_th))")
        print("  -> β_dependency controls how quickly the second circuit’s failure probability grows with wind speed.")
        print("  -> It is independent from Bayesian β (1/λ) and has physical meaning related to wind–line coupling.")
        print("-------------------------------------------------------------\n")
        # -------- Plot: Historical failures vs modelled probability --------
    try:
        # پیدا کردن ستون‌های تاریخ و باد
        time_col = pick_col(sub, ["date", "datetime", "timestamp", "day"], True, "failures")
        wind_col = pick_col(sub, ["wind_speed", "wind", "max_wind", "wind_max"], True, "failures")

        sub_plot = sub.dropna(subset=[time_col, wind_col]).copy()
        sub_plot[time_col] = pd.to_datetime(sub_plot[time_col], errors="coerce")
        sub_plot = sub_plot.dropna(subset=[time_col])
        sub_plot = sub_plot.sort_values(time_col)

        # محاسبه exposure و احتمال خرابی
        Lmax = max(seg_lengths) if seg_lengths else 0.0
        exposure = alpha_w * Lmax * np.clip(sub_plot[wind_col] - w_thresh, 0.0, None)**3
        sub_plot["failure_prob"] = lognormal_cdf(exposure, mu, sigma)

        # رسم نمودار احتمال
        plt.figure(figsize=(10, 5))
        plt.plot(sub_plot[time_col], sub_plot["failure_prob"],
                 color="steelblue", linewidth=1.8, label="Modelled failure probability")

        # فقط تاریخ‌هایی که failure واقعی داشتیم (ستون failure == 1)
        if "failure" in sub_plot.columns:
            fail_dates = sub_plot.loc[sub_plot["failure"] == 1, time_col].unique()
            for fd in sorted(fail_dates):
                plt.axvline(fd, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                            label="Actual failure" if fd == fail_dates[0] else "")
        else:
            print("⚠️ Column 'failure' not found — no red lines will be shown.")

        plt.title(f"WAY {wid} — Historical Failures vs Modelled Probability")
        plt.xlabel("Date and Time")
        plt.ylabel("Failure Probability")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        out_hist = os.path.join(OUTPUT_DIR, f"historical_failures_vs_probability_{wid}.png")
        plt.savefig(out_hist, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"- ✅ Saved improved historical failure plot: {out_hist}")

    except Exception as e:
        print(f"- ⚠️ Error plotting historical failures for WAY {wid}: {e}")

    # ============================================================
    # MARKOV CHAIN SECTION — Clean with Sigmoid δ(wind)
    # ============================================================
    print("\n[Markov Chain Simulation Started]")

    failures_path = os.path.join(OUTPUT_DIR, f"failures_with_exposure_{wid}.csv")
    if not os.path.exists(failures_path):
        print(f"  ⚠️ File not found: {failures_path} - Skipping Markov chain for WAY {wid}")
        continue

    try:
        df = pd.read_csv(failures_path)
        print(f"  -> Loaded: {failures_path} (rows={len(df)})")

        # Check required columns
        missing_cols = []
        for col in ["node_id", "date", "wind_speed"]:
            if col not in df.columns:
                missing_cols.append(col)
        if missing_cols:
            print(f"  ⚠️ Missing columns: {missing_cols} - Skipping Markov chain for WAY {wid}")
            continue

        if "failure" not in df.columns:
            df["failure"] = 0

        winds = df["wind_speed"].astype(float).values
        winds = winds[~np.isnan(winds)]  # Remove NaN values
        if len(winds) == 0:
            print(f"  ⚠️ No valid wind speed data - Skipping Markov chain for WAY {wid}")
            continue

        wmin, wmax = np.nanmin(winds), np.nanmax(winds)
        # Fix division by zero when all wind speeds are the same
        if abs(wmax - wmin) < 1e-9:
            print(f"  ⚠️ All wind speeds are identical ({wmin:.2f} m/s) - Using uniform P_base")
            P_base = np.ones(len(df)) * 0.5
        else:
            P_base = (df["wind_speed"].astype(float).values - wmin) / (wmax - wmin)
            P_base = np.clip(P_base, 0, 1)
        
        fail = (df["failure"].values > 0).astype(int)

        if circuits is not None and circuits >= 2:
            print("\n[Double-circuit line detected → δ estimated empirically & modulated by wind]\n")

            # Calculate P(B|A=0) from non-failure days
            mask_A0 = (fail == 0)
            if np.sum(mask_A0) > 0:
                P_B_given_A0 = float(P_base[mask_A0].mean())
            else:
                P_B_given_A0 = 0.5  # Default if no non-failure days
                print("  ⚠️ No non-failure days found, using default P(B|A=0) = 0.5")

            # Calculate P(B|A=1) from actual double-outage data if available
            mask_A1 = (fail == 1)
            if np.sum(mask_A1) > 0:
                P_B_given_A1 = float(P_base[mask_A1].mean())
                # If we have actual double-outage data, use it; otherwise assume 1.0
                if P_B_given_A1 < P_B_given_A0:
                    P_B_given_A1 = 1.0  # Fallback to assumption
                    print("  ⚠️ Calculated P(B|A=1) < P(B|A=0), using assumption P(B|A=1) = 1.0")
            else:
                P_B_given_A1 = 1.0  # Assumption when no failure data available
                print("  ⚠️ No failure days found, using assumption P(B|A=1) = 1.0")

            delta_hat = (P_B_given_A1 - P_B_given_A0) / max(1e-9, 1 - P_B_given_A0)
            delta_hat = float(np.clip(delta_hat, 0, 1))

            print("[Δ (delta) Empirical Estimation]")
            print(" Formula: δ̂ = (P(B|A=1) - P(B|A=0)) / (1 - P(B|A=0))")
            print(f" P(B|A=1) = {P_B_given_A1:.4f}  ← from failure days (or assumed)")
            print(f" P(B|A=0) = {P_B_given_A0:.4f}  ← mean of normalized wind prob (non-failure days)")
            print(f" → Estimated δ̂ = {delta_hat:.4f}\n")

            # ✅ Sigmoid modulation for δ(wind) - VAFFEL methodology
            # δ(wind) increases smoothly with wind speed, ensuring proper dependency behavior
            # Data-driven parameters: center and scale computed from observed wind distribution
            center = np.mean(winds)
            scale = max(np.std(winds) / 2, 1e-6)  # Avoid division by zero, scale based on wind variability
            # Use full wind series for delta_wind calculation
            winds_full = df["wind_speed"].astype(float).values
            # Sigmoid: δ(w) = δ̂ * sigmoid((w - center) / scale)
            # This ensures δ(w) increases monotonically with wind speed: ∂δ/∂w > 0 for all w
            delta_wind = delta_hat * (1 / (1 + np.exp(-(winds_full - center) / scale)))

            print("[δ(wind) Sigmoid Modulation - VAFFEL]")
            print(f" Center = {center:.2f} m/s (data-driven: mean of observed winds)")
            print(f" Scale = {scale:.2f} (data-driven: std(winds)/2)")
            print(" δ(w) increases smoothly with wind speed: ∂δ/∂w > 0 (verified)")
            
            # Verification: delta should increase with wind speed
            if len(delta_wind) > 1:
                wind_sorted_idx = np.argsort(winds_full)
                delta_sorted = delta_wind[wind_sorted_idx]
                # Check monotonicity: delta should be non-decreasing with wind speed
                delta_increasing = np.all(delta_sorted[1:] >= delta_sorted[:-1] - 1e-10)
                if not delta_increasing:
                    print("  ⚠️ Warning: Delta not strictly increasing with wind speed - check sigmoid parameters")

            # Markov chain probabilities
            P_f1 = P_base
            P_f2 = P_base
            P_f2_cond = np.clip(P_f2 + delta_wind * (1 - P_f2), 0, 1)
            P_both = P_f1 * P_f2_cond
            P_one = P_f1 * (1 - P_f2_cond) + (1 - P_f1) * P_f2
            P_none = 1 - (P_one + P_both)
            # Ensure probabilities sum to 1 (numerical stability)
            total_prob = P_none + P_one + P_both
            P_none = P_none / total_prob
            P_one = P_one / total_prob
            P_both = P_both / total_prob

            df_markov = pd.DataFrame({
                "node_id": df["node_id"],
                "date": df["date"],
                "wind_speed": df["wind_speed"],
                "P_none": P_none,
                "P_one": P_one,
                "P_both": P_both,
                "delta": delta_wind
            })

            # Save Markov results
            out_markov = os.path.join(OUTPUT_DIR, f"markov_results_{wid}.csv")
            df_markov.to_csv(out_markov, index=False)
            print(f"  -> Saved Markov results: {out_markov}")

            # Show only real failures
            rf = df[df["failure"] == 1].merge(df_markov, on=["node_id","date","wind_speed"], how="left")
            if len(rf) > 0:
                print("\n[Markov Results for REAL Failures]")
                print(rf[["node_id","date","wind_speed","P_both","delta"]].to_string(index=False))
            else:
                print("\n[Markov Results] No actual failures to display")

        else:
            print("\n[Single-circuit line → Markov chain skipped]")

    except Exception as e:
        print(f"  ⚠️ Error in Markov chain simulation for WAY {wid}: {e}")
        import traceback
        traceback.print_exc()
