from os.path import abspath, dirname, join
import scipy.constants as co
import matplotlib.pyplot as plt
from bolos import parser, solver, grid

mechanism = join(dirname(abspath(__file__)),  "itikawa-2009-O2.txt")
print(mechanism)


def main():
    # Use a linear grid from 0 to 60 eV with 500 intervals.
    gr = grid.LinearGrid(0.001, 60, 100)
    # gr = grid.GeometricGrid(0.1, 60, 100)

    # Initiate the solver instance
    bsolver = solver.BoltzmannSolver(gr)

    # Parse the cross-section file in BOSIG+ format and load it into the
    # solver.
    with open(mechanism, "r", encoding="utf-8") as fp:
        bsolver.load_collisions(parser.parse(fp))

    # Set the conditions.  And initialize the solver
    T = 1000  # Gas temperature, K
    P = 101325  # pression, Pa
    ND = P / co.k / T  # densité, m^-3
    bsolver.target['O2'].density = 1.0  # molar fraction, -
    bsolver.kT = T * co.k / co.eV  # Gas temperature, eV
    E = 1e5  # Electric field, V
    bsolver.EN = E / ND  # * solver.TOWNSEND  # Reduced electric field, in V.m^2
    bsolver.init()

    # Start with Maxwell EEDF as initial guess.  Here we are starting with
    # with an electron temperature of 2 eV
    f0 = bsolver.maxwell(2.0)

    # Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
    f1 = bsolver.converge(f0, maxn=200, rtol=1e-5)

    # Calculate the properties.
    print(f"mobility = {bsolver.mobility(f1) / ND:.3f} 1/m/V/s")
    print(f"diffusion = {bsolver.diffusion(f1) / ND:.3f} 1/m/s")
    print(f"average energy = {bsolver.mean_energy(f1):.3f} eV")
    # print("electron temperature = %.3f K" % bsolver.electron_temperature(f1))

    plt.plot(bsolver.cenergy, f0, label="f0")
    plt.plot(bsolver.cenergy, f1, label="f1")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
