import numpy as np

class Scanner:
    """Monolithic PET scanner geometry."""
    def __init__(self):
        # Detector dimensions
        self.inner_radius = 235.422  # mm
        self.outer_radius = 278.296  # mm
        self.axial_length = 296.0  # mm
        
        # Derived
        self.z_bottom = -self.axial_length / 2
        self.z_top = self.axial_length / 2
        
        # Material
        self.detector_material = "G4_BGO"


class Phantom:
    """NEMA IEC sphere insert phantom."""
    def __init__(self):
        self.radius = 80.0  # mm
        self.height = 70.0  # mm
        self.sphere_diameters = [10, 13, 17, 22, 28, 37]  # mm
        self.sphere_orbit_radius = 45.0  # mm
        self.material = "G4_WATER"
        
        # Activity ratio (sphere:background)
        self.activity_ratio = 4.0
        
        # Computed sphere positions (x, y, z, radius)
        self.sphere_positions = []
        for i, d in enumerate(self.sphere_diameters):
            angle = i * (2 * np.pi / 6)
            x = self.sphere_orbit_radius * np.cos(angle)
            y = self.sphere_orbit_radius * np.sin(angle)
            self.sphere_positions.append((x, y, 0.0, d / 2))


class Reconstruction:
    """Default reconstruction parameters."""
    voxel_size = 2.0   # mm
    fov = 200.0        # mm
    n_iterations = 2
    n_subsets = 8


class Attenuation:
    """Attenuation coefficients at 511 keV (NIST XCOM)."""
    mu_water = 0.0096  # mm⁻¹
    mu_air = 1.05e-7   # mm⁻¹


# Pre-instantiated defaults
scanner = Scanner()
phantom = Phantom()
reconstruction = Reconstruction()