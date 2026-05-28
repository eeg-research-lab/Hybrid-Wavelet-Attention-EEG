# ============================================================
# master_train.py
# FINAL MASTER TRAINING CONTROLLER
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
print(" FINAL MASTER TRAINING PIPELINE ")
print_bar()

pipeline_start = time.time()

# ============================================================
# WAVELET TRAINING
# ============================================================

print_bar()
print(" STARTING WAVELET TRAINING ")
print_bar()

wavelet_status = os.system(
    "python train_wavelet.py"
)

if wavelet_status != 0:

    raise RuntimeError(
        "Wavelet training failed."
    )

print_bar()
print(" WAVELET TRAINING COMPLETE ")
print_bar()

# ============================================================
# HYBRID TRAINING
# ============================================================

print_bar()
print(" STARTING HYBRID TRAINING ")
print_bar()

hybrid_status = os.system(
    "python train_hybrid.py"
)

if hybrid_status != 0:

    raise RuntimeError(
        "Hybrid training failed."
    )

print_bar()
print(" HYBRID TRAINING COMPLETE ")
print_bar()

# ============================================================
# FINAL SUMMARY
# ============================================================

pipeline_time = (
    time.time() - pipeline_start
) / 3600

print_bar()
print(" MASTER TRAINING COMPLETE ")
print_bar()

print(
    f"\nTotal Pipeline Time: "
    f"{pipeline_time:.2f} hours"
)

print_bar()