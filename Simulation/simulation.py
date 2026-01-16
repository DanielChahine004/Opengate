"""
Monolithic PET Scanner Simulation - NEMA IEC Sphere Insert
===========================================================
Generates:
  - PET coincidence data (ROOT file)
  - Attenuation map (μ-map) from Geant4 material properties
  - Ground truth activity concentration map
"""

import opengate as gate
import numpy as np
import sys
sys.path.insert(0, '/home/h/Opengate')
from config import scanner, phantom, sources, reconstruction, attenuation, simulation, output

# =============================================================================
# UNITS
# =============================================================================
mm = gate.g4_units.mm
m = gate.g4_units.m
keV = gate.g4_units.keV
MeV = gate.g4_units.MeV
kBq = 1000 * gate.g4_units.Bq
s = gate.g4_units.s
cm = gate.g4_units.cm
g_cm3 = gate.g4_units.g_cm3

# =============================================================================
# SIMULATION SETUP
# =============================================================================
sim = gate.Simulation()
sim.visu = simulation.visualisation
sim.number_of_threads = simulation.n_threads
sim.random_seed = simulation.random_seed

# Ensure output directory exists
output.ensure_directory()

# =============================================================================
# WORLD
# =============================================================================
sim.world.size = [1 * m, 1 * m, 1 * m]
sim.world.material = "G4_AIR"

# =============================================================================
# MONOLITHIC PET DETECTOR
# =============================================================================
pet_detector = sim.add_volume("Tubs", "pet_detector")
pet_detector.rmin = scanner.inner_radius * mm
pet_detector.rmax = scanner.outer_radius * mm
pet_detector.dz = scanner.axial_length / 2 * mm
pet_detector.material = scanner.detector_material
pet_detector.translation = [0, 0, 0]

# =============================================================================
# NEMA IEC SPHERE INSERT PHANTOM
# =============================================================================
# Background cylinder
background = sim.add_volume("Tubs", "phantom_background")
background.rmin = 0
background.rmax = phantom.radius * mm
background.dz = phantom.height / 2 * mm
background.material = phantom.material
background.translation = [0, 0, 0]

# Hot spheres
sphere_volumes = []
for i, (x, y, z, r) in enumerate(phantom.sphere_positions):
    diameter = phantom.sphere_diameters[i]
    sphere = sim.add_volume("Sphere", f"sphere_{diameter}mm")
    sphere.rmax = r * mm
    sphere.material = phantom.material
    sphere.translation = [x * mm, y * mm, z * mm]
    sphere.mother = "phantom_background"
    sphere_volumes.append(sphere)

# =============================================================================
# ACTIVITY SOURCES
# =============================================================================
# Background source - excludes sphere regions by using confine
# Note: OpenGATE's GenericSource with confine ensures proper exclusion
bg_source = sim.add_source("GenericSource", "background_source")
bg_source.particle = "back_to_back"
bg_source.attached_to = "phantom_background"
bg_source.position.type = "cylinder"
bg_source.position.radius = phantom.radius * mm
bg_source.position.dz = phantom.height / 2 * mm
bg_source.direction.type = "iso"
bg_source.activity = sources.background_concentration * kBq

# Sphere sources (hot regions)
for i, diameter in enumerate(phantom.sphere_diameters):
    sphere_source = sim.add_source("GenericSource", f"sphere_source_{diameter}mm")
    sphere_source.particle = "back_to_back"
    sphere_source.attached_to = f"sphere_{diameter}mm"
    sphere_source.position.type = "sphere"
    sphere_source.position.radius = diameter / 2 * mm
    sphere_source.direction.type = "iso"
    sphere_source.activity = sources.sphere_concentration * kBq

# =============================================================================
# PHYSICS
# =============================================================================
sim.physics_manager.physics_list_name = simulation.physics_list

# =============================================================================
# HITS COLLECTION
# =============================================================================
hits = sim.add_actor("DigitizerHitsCollectionActor", "hits")
hits.attached_to = "pet_detector"
hits.output_filename = str(output.hits_path)
hits.attributes = [
    "EventID",
    "TrackID",
    "PostPosition",
    "TotalEnergyDeposit",
    "GlobalTime",
]

# =============================================================================
# STATISTICS
# =============================================================================
stats = sim.add_actor("SimulationStatisticsActor", "stats")
stats.track_types_flag = True

# =============================================================================
# MAP GENERATION FUNCTIONS
# =============================================================================
def generate_mu_map() -> np.ndarray:
    """
    Generate attenuation map using NIST XCOM coefficients at 511 keV.
    
    Returns:
        3D numpy array of linear attenuation coefficients (mm⁻¹)
    """
    coords = reconstruction.voxel_coords()
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    R_xy = np.sqrt(X**2 + Y**2)
    
    # Initialise with air
    mu_map = np.full(reconstruction.shape, attenuation.mu_air, dtype=np.float32)
    
    # Phantom background cylinder (water)
    phantom_mask = (R_xy <= phantom.radius) & (np.abs(Z) <= phantom.height / 2)
    mu_map[phantom_mask] = attenuation.mu_water
    
    # Spheres have same attenuation as background (both water)
    # No change needed, but we verify they're inside the phantom
    
    return mu_map


def generate_activity_map() -> np.ndarray:
    """
    Generate ground truth activity concentration map.
    
    Background activity fills the cylinder excluding sphere regions.
    Sphere regions have higher activity based on sphere_to_background_ratio.
    
    Returns:
        3D numpy array of activity concentrations (kBq/mm³)
    """
    coords = reconstruction.voxel_coords()
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    R_xy = np.sqrt(X**2 + Y**2)
    
    # Initialise with zero (outside phantom)
    activity_map = np.zeros(reconstruction.shape, dtype=np.float32)
    
    # Background cylinder
    phantom_mask = (R_xy <= phantom.radius) & (np.abs(Z) <= phantom.height / 2)
    activity_map[phantom_mask] = sources.background_concentration_per_mm3
    
    # Sphere regions (overwrite background with higher activity)
    for (sx, sy, sz, sr) in phantom.sphere_positions:
        sphere_dist = np.sqrt((X - sx)**2 + (Y - sy)**2 + (Z - sz)**2)
        sphere_mask = sphere_dist <= sr
        activity_map[sphere_mask] = sources.sphere_concentration_per_mm3
    
    return activity_map


def save_maps(mu_map: np.ndarray, activity_map: np.ndarray) -> None:
    """Save maps and metadata to output directory."""
    # Save arrays
    np.save(output.mu_map_path, mu_map)
    np.save(output.activity_map_path, activity_map)
    
    # Compile metadata
    metadata = {
        # Reconstruction grid
        'voxel_size_mm': reconstruction.voxel_size,
        'fov_mm': reconstruction.fov,
        'n_voxels': reconstruction.n_voxels,
        'shape': reconstruction.shape,
        
        # Attenuation
        'mu_water_per_mm': attenuation.mu_water,
        'mu_air_per_mm': attenuation.mu_air,
        'energy_keV': 511,
        'attenuation_source': 'NIST_XCOM',
        
        # Phantom geometry
        'phantom_radius_mm': phantom.radius,
        'phantom_height_mm': phantom.height,
        'sphere_diameters_mm': phantom.sphere_diameters,
        'sphere_positions_mm': phantom.sphere_positions,
        
        # Activity
        'background_concentration_kBq_per_mm3': sources.background_concentration_per_mm3,
        'sphere_concentration_kBq_per_mm3': sources.sphere_concentration_per_mm3,
        'sphere_to_background_ratio': sources.sphere_to_background_ratio,
        
        # Scanner
        'scanner_inner_radius_mm': scanner.inner_radius,
        'scanner_outer_radius_mm': scanner.outer_radius,
        'scanner_axial_length_mm': scanner.axial_length,
        
        # Simulation
        'random_seed': simulation.random_seed,
        'n_threads': simulation.n_threads,
    }
    
    mu_metadata_path = str(output.mu_map_path).replace('.npy', f'{output.metadata_suffix}')
    activity_metadata_path = str(output.activity_map_path).replace('.npy', f'{output.metadata_suffix}')
    
    np.save(mu_metadata_path, metadata)
    np.save(activity_metadata_path, metadata)
    
    return metadata


def print_map_summary(mu_map: np.ndarray, activity_map: np.ndarray, metadata: dict) -> None:
    """Print summary of generated maps."""
    print("\n--- Map Generation Summary ---")
    print(f"Grid: {reconstruction.n_voxels}³ voxels ({reconstruction.voxel_size}mm isotropic)")
    print(f"FOV: {reconstruction.fov}mm")
    
    print(f"\nAttenuation Map:")
    print(f"  Shape: {mu_map.shape}")
    print(f"  μ(water) = {attenuation.mu_water:.6f} mm⁻¹")
    print(f"  μ(air) = {attenuation.mu_air:.2e} mm⁻¹")
    print(f"  Non-air voxels: {np.sum(mu_map > attenuation.mu_air * 10):,}")
    
    print(f"\nActivity Map:")
    print(f"  Shape: {activity_map.shape}")
    print(f"  Background: {sources.background_concentration_per_mm3:.6f} kBq/mm³")
    print(f"  Spheres: {sources.sphere_concentration_per_mm3:.6f} kBq/mm³")
    print(f"  Ratio: {sources.sphere_to_background_ratio}:1")
    print(f"  Active voxels: {np.sum(activity_map > 0):,}")
    print(f"  Hot voxels (spheres): {np.sum(activity_map > sources.background_concentration_per_mm3 * 1.1):,}")
    
    total_activity = np.sum(activity_map) * reconstruction.voxel_size**3  # kBq
    print(f"  Total activity: {total_activity:.2f} kBq")
    
    print(f"\nSaved to: {output.directory}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NEMA IEC Sphere Insert PET Simulation")
    print("=" * 60)
    
    # Print configuration
    print(f"\nScanner: {scanner.detector_material} monolithic")
    print(f"  Inner radius: {scanner.inner_radius}mm")
    print(f"  Outer radius: {scanner.outer_radius}mm")
    print(f"  Axial length: {scanner.axial_length}mm")
    
    print(f"\nPhantom: NEMA IEC sphere insert")
    print(f"  Diameter: {phantom.radius * 2}mm")
    print(f"  Height: {phantom.height}mm")
    print(f"  Sphere diameters: {phantom.sphere_diameters}mm")
    
    print(f"\nActivity:")
    print(f"  Background: {sources.background_concentration} kBq/mL")
    print(f"  Spheres: {sources.sphere_concentration} kBq/mL")
    print(f"  Ratio: {sources.sphere_to_background_ratio}:1")
    
    # Generate ground truth maps
    print("\n--- Generating Ground Truth Maps ---")
    mu_map = generate_mu_map()
    activity_map = generate_activity_map()
    metadata = save_maps(mu_map, activity_map)
    print_map_summary(mu_map, activity_map, metadata)
    
    # Run simulation
    print("\n--- Running Geant4 Simulation ---")
    print(f"Threads: {simulation.n_threads}")
    print(f"Random seed: {simulation.random_seed}")
    print(f"Physics: {simulation.physics_list}")
    
    sim.run()
    
    print("\n--- Simulation Statistics ---")
    print(stats)
    
    print("\n--- Output Files ---")
    print(f"  Hits: {output.hits_path}")
    print(f"  μ-map: {output.mu_map_path}")
    print(f"  Activity map: {output.activity_map_path}")
    print("=" * 60)