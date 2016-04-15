
# no biased potential

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from ANN import *

############################ PARAMETERS BEGIN ###############################################################
record_interval = 500
total_number_of_steps = 40000

input_pdb_file_of_molecule = './dependency/alanine_dipeptide.pdb'
force_field_file = 'amber99sb.xml'
water_field_file = 'amber99_obc.xml'

pdb_reporter_file = './dependency/unbiased_output.pdb'
state_data_reporter_file = './dependency/unbiased_report.txt'


############################ PARAMETERS END ###############################################################


pdb = PDBFile(input_pdb_file_of_molecule) 
forcefield = ForceField(force_field_file) # without water
system = forcefield.createSystem(pdb.topology,  nonbondedMethod=NoCutoff, \
                                 constraints=AllBonds)  # what does it mean by topology? 

platform = Platform.getPlatformByName('Reference')
platform.loadPluginsFromDirectory('.')  # load the plugin from the current directory

force = ANN_Force()

force.set_layer_types(['Linear', 'Tanh'])
list_of_index_of_atoms_forming_dihedrals = [[2,5,7,9],
											[5,7,9,15],
											[7,9,15,17],
											[9,15,17,19]]

force.set_list_of_index_of_atoms_forming_dihedrals(list_of_index_of_atoms_forming_dihedrals)
force.set_num_of_nodes([8,8,8])
force.set_potential_center([tanh(0.9), tanh(0.44), tanh(-0.6), tanh(0.8),tanh(0.9), tanh(0.44), tanh(-0.6), tanh(0.8)])
force.set_force_constant(5000)
force.set_values_of_biased_nodes([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
force.set_coeffients_of_connections([
									[1,0,0,0,0,0,0,0,
									0,1,0,0,0,0,0,0,
									0,0,1,0,0,0,0,0,
									0,0,0,1,0,0,0,0,
									0,0,0,0,1,0,0,0,
									0,0,0,0,0,1,0,0,
									0,0,0,0,0,0,1,0,
									0,0,0,0,0,0,0,1,
									],
									[1,0,0,0,0,0,0,0,
									0,1,0,0,0,0,0,0,
									0,0,1,0,0,0,0,0,
									0,0,0,1,0,0,0,0,
									0,0,0,0,1,0,0,0,
									0,0,0,0,0,1,0,0,
									0,0,0,0,0,0,1,0,
									0,0,0,0,0,0,0,1,
									]
									])

system.addForce(force)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)


simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter(pdb_reporter_file, record_interval))
simulation.reporters.append(StateDataReporter(state_data_reporter_file, record_interval, step=True, potentialEnergy=True, temperature=True))
simulation.step(total_number_of_steps)

print('Done!')
