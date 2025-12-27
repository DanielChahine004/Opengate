"""
Monolithic PET Scanner Simulation - Radioactive Cube
=====================================================
"""

import opengate as gate

# =============================================================================
# SIMULATION SETUP
# =============================================================================
sim = gate.Simulation()
sim.visu = False
sim.number_of_threads = 12
sim.random_seed = 42

# Units
mm = gate.g4_units.mm
m = gate.g4_units.m
keV = gate.g4_units.keV
kBq = 1000 * gate.g4_units.Bq
s = gate.g4_units.s

# =============================================================================
# WORLD
# =============================================================================
sim.world.size = [1 * m, 1 * m, 1 * m]
sim.world.material = "G4_AIR"

# =============================================================================
# MONOLITHIC PET DETECTOR
# =============================================================================
inner_radius = 235.422 * mm / 2
outer_radius = 278.296 * mm / 2
axial_length = 296 * mm

pet_detector = sim.add_volume("Tubs", "pet_detector")
pet_detector.rmin = inner_radius
pet_detector.rmax = outer_radius
pet_detector.dz = axial_length / 2
pet_detector.material = "G4_BGO"
pet_detector.translation = [0, 0, 0]

# =============================================================================
# RADIOACTIVE CUBE (F-18 in water)
# =============================================================================
cube = sim.add_volume("Box", "radioactive_cube")
cube.size = [50 * mm, 50 * mm, 50 * mm]  # 5cm cube
cube.material = "G4_WATER"
cube.translation = [0, 0, 0]

# F-18 source filling the cube
source = sim.add_source("GenericSource", "f18_cube")
source.particle = "back_to_back"
source.attached_to = "radioactive_cube"
source.position.type = "box"
source.position.size = [50 * mm, 50 * mm, 50 * mm]
source.direction.type = "iso"
source.activity = 100 * kBq # determines how long the simulation runs

# =============================================================================
# PHYSICS
# =============================================================================
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"

# =============================================================================
# HITS COLLECTION
# =============================================================================
hits = sim.add_actor("DigitizerHitsCollectionActor", "hits")
hits.attached_to = "pet_detector"
hits.output_filename = "/home/h/Opengate/Simulation/Outputs/pet_hits.root"
hits.attributes = [
    "EventID",
    "TrackID",
    "PostPosition",
    "TotalEnergyDeposit",
    "GlobalTime",
]

# =============================================================================
# STATS
# =============================================================================
stats = sim.add_actor("SimulationStatisticsActor", "stats")
stats.track_types_flag = True

# =============================================================================
# RUN
# =============================================================================

print("Running simulation...")
sim.run()
print(stats)