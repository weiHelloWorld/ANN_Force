from simtk.openmm import System
from ANN import *

system = System()
a = ANN_Force()
a.set_list_of_index_of_atoms_forming_dihedrals_from_index_of_backbone_atoms([1,2,3, 4, 5, 6])
system.addForce(a)
print "Python wrapper test passed!"
