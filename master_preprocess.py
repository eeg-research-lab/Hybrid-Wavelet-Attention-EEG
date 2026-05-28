# ============================================================
# master_preprocess.py
# FINAL MASTER PREPROCESS CONTROLLER
# ============================================================

import os
import time

# ============================================================
# TERMINAL BAR
# ============================================================

def print_bar():

    print("=" * 60)

# ============================================================
# START
# ============================================================

print_bar()
print(" FINAL MASTER PREPROCESS PIPELINE ")
print_bar()

pipeline_start = time.time()

# ============================================================
# WAVELET
# ============================================================

print_bar()
print(" STARTING WAVELET PREPROCESSING ")
print_bar()

wavelet_status = os.system(
    "python preprocess_wavelet.py"
)

if wavelet_status != 0:

    raise RuntimeError(
        "Wavelet preprocessing failed."
    )

print_bar()
print(" WAVELET PREPROCESSING COMPLETE ")
print_bar()

# ============================================================
# HYBRID
# ============================================================

print_bar()
print(" STARTING HYBRID PREPROCESSING ")
print_bar()

hybrid_status = os.system(
    "python preprocess_hybrid.py"
)

if hybrid_status != 0:

    raise RuntimeError(
        "Hybrid preprocessing failed."
    )

print_bar()
print(" HYBRID PREPROCESSING COMPLETE ")
print_bar()

# ============================================================
# FINAL SUMMARY
# ============================================================

pipeline_time = (
    time.time() - pipeline_start
) / 3600

print_bar()
print(" MASTER PREPROCESS COMPLETE ")
print_bar()

print(
    f"\nTotal Pipeline Time: "
    f"{pipeline_time:.2f} hours"
)

print_bar()