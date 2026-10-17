import sys, os
sys.path.insert(0, ".")
from src.models.xgboost_model import XGBoostPitModel
import numpy as np

model = XGBoostPitModel()
model.load(os.path.join("models", "xgboost"))

# Fit quadratic to each compound's profile data (life >= 3, ignore outlap)
# y = a*life^2 + b*life + c
# degradation_increase(L) = y(L) - y(L0) where L0 is start of stint
print("=== Quadratic fits to compound profiles ===")
for compound in ["SOFT", "MEDIUM", "HARD"]:
    profile = model.compound_profiles[compound]
    tl = profile["tyre_lives"]
    dg = profile["degradation"]
    # Filter: life >= 3 and life <= reasonable limit for compound
    max_life = {"SOFT": 20, "MEDIUM": 30, "HARD": 40}[compound]
    mask = (tl >= 3) & (tl <= max_life)
    x = tl[mask]
    y = dg[mask]
    
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    print(f"\n{compound}: y = {a:.5f}*L² + {b:.5f}*L + {c:.5f}")
    print(f"  (fitted on {len(x)} points, life range [{x[0]:.0f}, {x[-1]:.0f}])")
    
    # Compute degradation increase from baseline (life=3)
    baseline = a*3**2 + b*3 + c
    thresholds = {"SOFT": 1.2, "MEDIUM": 1.5, "HARD": 2.0}
    thresh = thresholds[compound]
    for life in [5, 10, 15, 20, 25, 30, 35, 40]:
        fitted = a*life**2 + b*life + c
        increase = max(fitted - baseline, 0)
        marker = " <-- PIT" if increase >= thresh and increase - (a*(2*life-1) + b) < thresh else ""
        print(f"  life={life:2d}: fitted={fitted:.3f}s  increase={increase:.3f}s{marker}")
        if marker:
            print(f"  → With threshold {thresh}s, pit at lap {life}")
            break
    else:
        # Find exact pit lap
        for life in range(3, 60):
            fitted = a*life**2 + b*life + c
            increase = max(fitted - baseline, 0)
            if increase >= thresh:
                print(f"  → With threshold {thresh}s, pit at lap {life} (increase={increase:.3f})")
                break

# Now test: what if we use the fitted coefficients directly in the strategy?
print("\n=== Strategy with fitted coefficients ===")
# For each compound, the "effective rate" for our formula deg = rate*(0.015*L² + 0.6*L)
# Match it to the fitted quadratic at the target pit lap
for compound in ["SOFT", "MEDIUM", "HARD"]:
    profile = model.compound_profiles[compound]
    tl = profile["tyre_lives"]
    dg = profile["degradation"]
    max_life = {"SOFT": 20, "MEDIUM": 30, "HARD": 40}[compound]
    mask = (tl >= 3) & (tl <= max_life)
    x = tl[mask]
    y = dg[mask]
    coeffs = np.polyfit(x, y, 2) 
    a, b, c = coeffs
    baseline = a*3**2 + b*3 + c
    
    # The fitted quadratic coefficients a and b describe the data-driven curve
    # Our formula: deg = rate * (0.015*L² + 0.6*L)  →  0.015*rate = a_data, 0.6*rate = b_data
    # Since those may not give same rate, use the increase at the target pit lap
    target_laps = {"SOFT": 15, "MEDIUM": 24, "HARD": 34}
    tgt = target_laps[compound]
    increase_at_target = max(a*tgt**2 + b*tgt + c - baseline, 0.01)
    thresh = {"SOFT": 1.2, "MEDIUM": 1.5, "HARD": 2.0}[compound]
    print(f"{compound}: increase at target life {tgt} = {increase_at_target:.3f}s (threshold={thresh}s)")
