"""
Post-Processing Script for Coincidence Detection (Polars version)
==================================================================
Processes raw hits from PET simulation to extract:
- First interaction points for each gamma
- Coincident pairs within timing window
- Output: (xyz1, xyz2) for each true coincidence

For training PointNet-based gamma localization networks.
"""

import polars as pl
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
COINCIDENCE_WINDOW_NS = 4.5  # ns
INPUT_FILE = "/home/h/Opengate/Simulation/Outputs/pet_hits.root"
OUTPUT_FILE = "/home/h/Opengate/Post_process/Outputs/coincidence_pairs.npz"


def load_hits(filename: str) -> pl.DataFrame:
    """Load hits from ROOT file into Polars DataFrame."""
    import uproot
    
    print(f"Loading hits from {filename}...")
    
    with uproot.open(filename) as f:
        # Find hits tree
        tree_name = next((k for k in f.keys() if "hits" in k.lower()), None)
        if tree_name is None:
            raise ValueError("Could not find hits tree in ROOT file")
        
        tree = f[tree_name]
        print(f"Tree: {tree_name}, Entries: {tree.num_entries}")
        
        # Load as dict of arrays, then to Polars
        arrays = tree.arrays(library="np")
        df = pl.DataFrame({k: v for k, v in arrays.items()})
    
    print(f"Loaded {len(df)} hits")
    print(f"Columns: {df.columns}")
    return df


def extract_first_interactions(hits: pl.DataFrame) -> pl.DataFrame:
    """Extract the first interaction point for each (EventID, TrackID)."""
    print("Extracting first interactions...")
    
    first_hits = (
        hits
        .sort("GlobalTime")
        .group_by(["EventID", "TrackID"])
        .first()
    )
    
    print(f"Found {len(first_hits)} first interactions from {len(hits)} total hits")
    return first_hits


def find_coincidences(first_hits: pl.DataFrame, time_window_ns: float = COINCIDENCE_WINDOW_NS) -> pl.DataFrame:
    """
    Find coincident gamma pairs using Polars.
    
    Strategy: self-join on EventID, filter for valid pairs.
    """
    print(f"Finding coincidences (window={time_window_ns} ns)...")
    
    # Rename columns for self-join
    df1 = first_hits.select([
        pl.col("EventID"),
        pl.col("TrackID").alias("TrackID1"),
        pl.col("PostPosition_X").alias("x1"),
        pl.col("PostPosition_Y").alias("y1"),
        pl.col("PostPosition_Z").alias("z1"),
        pl.col("TotalEnergyDeposit").alias("e1"),
        pl.col("GlobalTime").alias("t1"),
    ])
    
    df2 = first_hits.select([
        pl.col("EventID"),
        pl.col("TrackID").alias("TrackID2"),
        pl.col("PostPosition_X").alias("x2"),
        pl.col("PostPosition_Y").alias("y2"),
        pl.col("PostPosition_Z").alias("z2"),
        pl.col("TotalEnergyDeposit").alias("e2"),
        pl.col("GlobalTime").alias("t2"),
    ])
    
    # Self-join on EventID
    joined = df1.join(df2, on="EventID", how="inner")
    
    # Filter: different tracks, TrackID1 < TrackID2 (avoid duplicates)
    coincidences = (
        joined
        .filter(pl.col("TrackID1") < pl.col("TrackID2"))
        .with_columns([
            (pl.col("t1") - pl.col("t2")).abs().alias("time_diff")
        ])
        .filter(pl.col("time_diff") <= time_window_ns)
        .select([
            "EventID", "x1", "y1", "z1", "x2", "y2", "z2",
            "e1", "e2", "t1", "t2", "time_diff"
        ])
    )
    
    print(f"Found {len(coincidences)} valid coincidences")
    return coincidences


def compute_lor_statistics(df: pl.DataFrame) -> None:
    """Compute and print LOR statistics."""
    if len(df) == 0:
        print("No coincidences to analyze")
        return
    
    # Compute derived quantities
    stats = df.select([
        # LOR length
        ((pl.col("x1") - pl.col("x2")).pow(2) +
         (pl.col("y1") - pl.col("y2")).pow(2) +
         (pl.col("z1") - pl.col("z2")).pow(2)).sqrt().alias("lor_length"),
        # Radial positions
        (pl.col("x1").pow(2) + pl.col("y1").pow(2)).sqrt().alias("r1"),
        (pl.col("x2").pow(2) + pl.col("y2").pow(2)).sqrt().alias("r2"),
        # Z positions
        pl.col("z1"),
        pl.col("z2"),
        pl.col("time_diff"),
    ])
    
    print("\n" + "=" * 50)
    print("LOR STATISTICS")
    print("=" * 50)
    print(f"Number of coincidences: {len(df)}")
    
    lor = stats["lor_length"]
    print(f"LOR length: {lor.mean():.2f} ± {lor.std():.2f} mm")
    print(f"  Min: {lor.min():.2f} mm, Max: {lor.max():.2f} mm")
    
    r1, r2 = stats["r1"], stats["r2"]
    print(f"Radial position 1: {r1.mean():.2f} ± {r1.std():.2f} mm")
    print(f"Radial position 2: {r2.mean():.2f} ± {r2.std():.2f} mm")
    
    z1, z2 = stats["z1"], stats["z2"]
    print(f"Z range 1: [{z1.min():.2f}, {z1.max():.2f}] mm")
    print(f"Z range 2: [{z2.min():.2f}, {z2.max():.2f}] mm")
    
    td = stats["time_diff"]
    print(f"Time difference: {td.mean():.4f} ± {td.std():.4f} ns")


def save_training_data(df: pl.DataFrame, output_file: str = "coincidence_pairs.npz"):
    """Save as NPZ for direct NumPy/Numba use."""
    if len(df) == 0:
        print("No data to save")
        return
    
    xyz1 = df.select(["x1", "y1", "z1"]).to_numpy().astype(np.float32)
    xyz2 = df.select(["x2", "y2", "z2"]).to_numpy().astype(np.float32)
    
    np.savez(output_file, xyz1=xyz1, xyz2=xyz2)
    
    print(f"Saved {len(xyz1):,} LORs to {output_file}")


def main():
    print("=" * 60)
    print("PET COINCIDENCE POST-PROCESSING (Polars)")
    print("=" * 60)
    
    # Load
    hits = load_hits(INPUT_FILE)
    
    # Process
    first_hits = extract_first_interactions(hits)
    coincidences = find_coincidences(first_hits)
    
    if len(coincidences) > 0:
        compute_lor_statistics(coincidences)
        save_training_data(coincidences, output_file=OUTPUT_FILE)
    else:
        print("\nNo coincidences found.")


if __name__ == "__main__":
    main()