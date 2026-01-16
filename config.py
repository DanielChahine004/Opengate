"""
Configuration for Monolithic PET Scanner Simulation
====================================================
All parameters consolidated for reproducibility and parameter sweeps.
"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Scanner:
    """Monolithic PET scanner geometry."""
    inner_radius: float = 235.422  # mm
    outer_radius: float = 278.296  # mm
    axial_length: float = 296.0  # mm
    detector_material: str = "G4_BGO"
    
    @property
    def z_bottom(self) -> float:
        return -self.axial_length / 2
    
    @property
    def z_top(self) -> float:
        return self.axial_length / 2


@dataclass
class Phantom:
    """NEMA IEC sphere insert phantom."""
    radius: float = 80.0  # mm
    height: float = 70.0  # mm
    sphere_diameters: tuple[float, ...] = (10, 13, 17, 22, 28, 37)  # mm
    sphere_orbit_radius: float = 45.0  # mm
    material: str = "G4_WATER"
    
    @property
    def sphere_positions(self) -> list[tuple[float, float, float, float]]:
        """Compute sphere positions (x, y, z, radius) in mm."""
        positions = []
        n_spheres = len(self.sphere_diameters)
        for i, d in enumerate(self.sphere_diameters):
            angle = i * (2 * np.pi / n_spheres)
            x = self.sphere_orbit_radius * np.cos(angle)
            y = self.sphere_orbit_radius * np.sin(angle)
            positions.append((x, y, 0.0, d / 2))
        return positions


@dataclass
class Sources:
    """Activity source parameters."""
    background_concentration: float = 5.0  # kBq/mL (= kBq/cm³)
    sphere_to_background_ratio: float = 4.0
    
    @property
    def sphere_concentration(self) -> float:
        """Sphere activity concentration in kBq/mL."""
        return self.background_concentration * self.sphere_to_background_ratio
    
    @property
    def background_concentration_per_mm3(self) -> float:
        """Background concentration in kBq/mm³."""
        return self.background_concentration / 1000  # 1 mL = 1000 mm³
    
    @property
    def sphere_concentration_per_mm3(self) -> float:
        """Sphere concentration in kBq/mm³."""
        return self.sphere_concentration / 1000


@dataclass
class Reconstruction:
    """Reconstruction grid parameters."""
    voxel_size: float = 1.0  # mm (isotropic)
    fov: float = 200.0  # mm
    
    @property
    def n_voxels(self) -> int:
        """Number of voxels per dimension."""
        return int(self.fov / self.voxel_size)
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """3D grid shape."""
        n = self.n_voxels
        return (n, n, n)
    
    @property
    def half_fov(self) -> float:
        """Half FOV for coordinate calculations."""
        return self.fov / 2
    
    def voxel_coords(self) -> np.ndarray:
        """1D array of voxel center coordinates."""
        return np.linspace(
            -self.half_fov + self.voxel_size / 2,
            self.half_fov - self.voxel_size / 2,
            self.n_voxels
        )


@dataclass
class Attenuation:
    """Attenuation coefficients at 511 keV from NIST XCOM database."""
    # Mass attenuation coefficients (cm²/g)
    mu_rho_water: float = 0.09597
    mu_rho_air: float = 0.08712
    
    # Densities (g/cm³)
    rho_water: float = 1.000
    rho_air: float = 0.001205
    
    @property
    def mu_water(self) -> float:
        """Linear attenuation coefficient for water (mm⁻¹)."""
        return self.mu_rho_water * self.rho_water / 10  # cm⁻¹ to mm⁻¹
    
    @property
    def mu_air(self) -> float:
        """Linear attenuation coefficient for air (mm⁻¹)."""
        return self.mu_rho_air * self.rho_air / 10


@dataclass
class Simulation:
    """Simulation execution parameters."""
    n_threads: int = 12
    random_seed: int = 42
    run_time: float = 1.0  # seconds
    physics_list: str = "G4EmStandardPhysics_option4"
    visualisation: bool = False


@dataclass
class Output:
    """Output paths and settings."""
    directory: Path = field(default_factory=lambda: Path("/home/h/Opengate/Simulation/Outputs"))
    hits_filename: str = "pet_hits.root"
    mu_map_filename: str = "mu_map.npy"
    activity_map_filename: str = "activity_map.npy"
    metadata_suffix: str = "_metadata.npy"
    
    def __post_init__(self):
        if isinstance(self.directory, str):
            self.directory = Path(self.directory)
    
    @property
    def hits_path(self) -> Path:
        return self.directory / self.hits_filename
    
    @property
    def mu_map_path(self) -> Path:
        return self.directory / self.mu_map_filename
    
    @property
    def activity_map_path(self) -> Path:
        return self.directory / self.activity_map_filename
    
    def ensure_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        self.directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Default instances
# =============================================================================
scanner = Scanner()
phantom = Phantom()
sources = Sources()
reconstruction = Reconstruction()
attenuation = Attenuation()
simulation = Simulation()
output = Output()