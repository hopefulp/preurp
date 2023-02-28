import argparse
import os
from ase import Atoms, Atom, units
import ase.io
from ase.calculators.emt import EMT
from ase.build import fcc110
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from amp.utilities import Annealer

def ase_newversion():
    ver = ase.__version__
    ref = '3.21.0'
    if ver >= ref:
        return 1
    else:
        return 0

def generate_data(count, trajfile='emtmd.traj'):
    """Generates test or training data with a simple MD simulation."""
    pivot = 86
    traj = ase.io.Trajectory(trajfile, 'w')
    atoms = fcc110('Pt', (4, 6, 4), vacuum=15.)
    atoms.extend(Atoms([Atom('Cu', atoms[pivot].position + (0., 0., 2.5)),
                        Atom('Cu', atoms[pivot].position + (0., 0., 5.))]))
    #atoms.set_constraint(FixAtoms(indices=[0, 2]))
    ase.io.write('POSCAR', atoms, format='vasp')

    atoms.set_calculator(EMT())
    atoms.get_potential_energy()
    traj.write(atoms)
    if ase_newversion() :
        MaxwellBoltzmannDistribution(atoms, temperature_K = 300.)
    else:
        MaxwellBoltzmannDistribution(atoms, 300.*units.kB)

    dyn = VelocityVerlet(atoms, timestep = 1. * units.fs)
    dump_step=10
    count = int(count/dump_step)	
    for step in range(count - 1):
        dyn.run(dump_step)
        traj.write(atoms)
    
    return trajfile


def run_training(Lforce):
    
    trajfile = generate_data(1000)
    images = ase.io.Trajectory(trajfile)
    
    calc = Amp(descriptor=Gaussian(), model=NeuralNetwork(hiddenlayers=(10, 10, 10)))
    Annealer(calc=calc, images=images, Tmax=20, Tmin=1, steps=4000)
    convergence={}
    convergence['energy_rmse'] = 0.001
    if Lforce:
        convergence['force_rmse'] = 0.02
        calc.model.lossfunction = LossFunction(convergence=convergence, force_coefficient=0.04)
    else:
        calc.model.lossfunction = LossFunction(convergence=convergence)
    calc.train(images=images)

def main():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('-f', '--force', action='store_true', help='add force training')
    args = parser.parse_args()

    run_training(args.force)

if __name__ == '__main__':
    main()


