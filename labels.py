# ============================================================
# labels.py
# ============================================================

import os
import re
from datetime import datetime

# ============================================================
# DATASET ROOT
# ============================================================

DATASET_ROOT = r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database"

# ============================================================
# TRAIN PATIENTS
# ============================================================

TRAIN_PATIENTS = [
    "chb01",
    "chb02"
]

# ============================================================
# TEST / APP PATIENTS
# ============================================================

ALL_PATIENTS = [

    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
    "chb19",
    "chb20"

]

# ============================================================
# MANUAL SUMMARY FILE PATHS
# ============================================================

PATIENT_SUMMARIES = {

    "chb01":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb01\chb01-summary.txt",

    "chb02":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb02\chb02-summary.txt",

    "chb03":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb03\chb03-summary.txt",

    "chb04":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb04\chb04-summary.txt",

    "chb05":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb05\chb05-summary.txt",

    "chb06":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb06\chb06-summary.txt",

    "chb07":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb07\chb07-summary.txt",

    "chb08":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb08\chb08-summary.txt",

    "chb09":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb09\chb09-summary.txt",

    "chb10":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb10\chb10-summary.txt",

    "chb11":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb11\chb11-summary.txt",

    "chb12":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb12\chb12-summary.txt",

    "chb13":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb13\chb13-summary.txt",

    "chb14":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb14\chb14-summary.txt",

    "chb15":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb15\chb15-summary.txt",

    "chb16":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb16\chb16-summary.txt",

    "chb17":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb17\chb17-summary.txt",

    "chb18":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb18\chb18-summary.txt",

    "chb19":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb19\chb19-summary.txt",

    "chb20":
    r"C:\Users\jagad\Desktop\CHB-MIT Scalp EEG Database\chb20\chb20-summary.txt",

}

# ============================================================
# TERMINAL BAR
# ============================================================

def print_bar():

    print("=" * 60)

# ============================================================
# LOGGER
# ============================================================

def log(message):

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    print(f"[{current_time}] {message}")

# ============================================================
# START
# ============================================================

print_bar()
print(" EEG LABEL MANAGEMENT SYSTEM ")
print_bar()

log("Initializing label system...")

# ============================================================
# VALIDATE SUMMARY FILES
# ============================================================

for patient in ALL_PATIENTS:

    if patient not in PATIENT_SUMMARIES:

        raise ValueError(
            f"Summary path missing for {patient}"
        )

    summary_path = PATIENT_SUMMARIES[patient]

    if not os.path.exists(summary_path):

        raise ValueError(
            f"Summary file not found:\n{summary_path}"
        )

    log(f"Validated summary: {patient}")

# ============================================================
# CACHE SUMMARIES
# ============================================================

SUMMARY_CACHE = {}

print_bar()
log("Caching summary files...")
print_bar()

for patient, summary_path in PATIENT_SUMMARIES.items():

    with open(summary_path, "r") as f:

        SUMMARY_CACHE[patient] = f.read()

    log(f"Cached: {patient}")

# ============================================================
# GET EDF FILES
# ============================================================

def get_patient_edf_files(patient):

    patient_folder = os.path.join(
        DATASET_ROOT,
        patient
    )

    edf_files = sorted([

        f for f in os.listdir(patient_folder)

        if f.endswith(".edf")

    ])

    return edf_files

# ============================================================
# GET EDF PATH
# ============================================================

def get_edf_path(patient,
                 edf_file):

    return os.path.join(
        DATASET_ROOT,
        patient,
        edf_file
    )

# ============================================================
# GET SEIZURE RANGES
# ============================================================

def get_seizure_ranges(patient,
                       edf_file):

    summary_text = SUMMARY_CACHE[patient]

    sections = re.split(
        r"File Name:",
        summary_text
    )

    seizure_ranges = []

    for section in sections[1:]:

        lines = section.strip().split("\n")

        filename = (
            lines[0]
            .strip()
            .replace("\r", "")
        )

        if filename != edf_file:
            continue

        for i, line in enumerate(lines):

            if "Seizure Start Time" in line:

                start_sec = int(
                    re.findall(r"\d+", line)[0]
                )

                end_line = lines[i + 1]

                end_sec = int(
                    re.findall(r"\d+", end_line)[0]
                )

                seizure_ranges.append(
                    (
                        start_sec,
                        end_sec
                    )
                )

    return seizure_ranges

# ============================================================
# DEBUG VALIDATION
# ============================================================

if __name__ == "__main__":

    print_bar()
    print(" LABEL VALIDATION TEST ")
    print_bar()

    total_edfs = 0
    total_seizures = 0

    for patient in TRAIN_PATIENTS:

        print_bar()

        log(f"PATIENT: {patient}")

        print_bar()

        edf_files = get_patient_edf_files(
            patient
        )

        log(
            f"EDF Files Found: {len(edf_files)}"
        )

        total_edfs += len(edf_files)

        for idx, edf in enumerate(edf_files):

            seizure_ranges = get_seizure_ranges(
                patient,
                edf
            )

            seizure_count = len(seizure_ranges)

            total_seizures += seizure_count

            print(
                f"[{idx+1:03d}/{len(edf_files)}] "
                f"{edf:<20} "
                f"Seizures: {seizure_count}"
            )

            if seizure_count > 0:

                for s_idx, (
                    start,
                    end
                ) in enumerate(seizure_ranges):

                    duration = end - start

                    print(
                        f"     └── Seizure {s_idx+1}: "
                        f"{start}s → {end}s "
                        f"(Duration: {duration}s)"
                    )

    print_bar()
    print(" FINAL SUMMARY ")
    print_bar()

    log(
        f"Total Patients : "
        f"{len(TRAIN_PATIENTS)}"
    )

    log(
        f"Total EDF Files: "
        f"{total_edfs}"
    )

    log(
        f"Total Seizures : "
        f"{total_seizures}"
    )

    print_bar()
    log("LABEL SYSTEM VALIDATED SUCCESSFULLY")
    print_bar()