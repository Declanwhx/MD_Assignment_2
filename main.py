import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.constants as co
import re
from mpl_toolkits.mplot3d import Axes3D

######################### CONSTANTS #########################
# Boltzmann Constant
k_B = co.k  # [J/K]
# Lennard-Jones variables -> Depth of well and collisional diameter
epslj = 148 * co.k  # [J]
sigma = 3.73  # [Angstroms]
sigma_sq = sigma ** 2  # [Angstroms^2]
sigma_cu = sigma ** 3  # [Angstroms^3]
CH4_molar_mass = 16.04  # [g/mol]
CH4_molecule_mass = CH4_molar_mass * (1 / co.N_A)  # [g]
# Degrees of freedom
ndim = 3  # [-]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # FUNCTIONS # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def convertMassDensity(mass_density):
    """
    Function that takes in the mass density and converts it to molecule density

    :param mass_density: Mass density of system [kg/m^3]
    :return: Molecule density [1/Angstrom^3]
    """
    molecule_density = ((mass_density / (CH4_molar_mass * 1e-3)) * co.N_A) / (10 ** 30)
    return molecule_density


def rdf(xyz, LxLyLz, n_bins=100, r_range=(0.01, 10.0)):
    """
    Radial pair distribution function

    :param xyz: coordinates in xyz format per frame
    :param LxLyLz: box length in vector format
    :param n_bins: number of bins
    :param r_range: range on which to compute rdf r_0 to r_max
    :return:
    """

    """
    Initializing histogram
    g_r is the array containing the no. of variables that fall within the respective bins
    edges is the array containing the range for each bin
    
    g_r[0] set as zero to prevent central molecule from being recorded as an overlap
    casting float type to each variable in array
    """
    g_r, edges = np.histogram([0], bins=n_bins, range=r_range)
    g_r[0] = 0
    g_r = g_r.astype(np.float64)
    N = 0

    for i, xyz_i in enumerate(xyz):
        # Building vertical array excluding molecule i for each loop
        xyz_j = np.vstack([xyz[:i], xyz[i + 1:]])
        # Distances between molecule i and molecules j != i in xyz array for each pair
        d = np.abs(xyz_i - xyz_j)
        # PBC revert
        d = np.where(d > 0.5 * LxLyLz, LxLyLz - d, d)
        # Radial distance
        d = np.sqrt(np.sum(d ** 2, axis=-1))
        # print(d)
        temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
        # print(temp_g_r)
        # Updating bin
        g_r += temp_g_r
        N = i + 1

    # Overall molecular density [1/Angstrom^3] N/V
    rho = N / np.prod(LxLyLz)
    # Calculating dist r by taking mid-dr
    r = 0.5 * (edges[1:] + edges[:-1])
    # Volume in dr
    V = 4. / 3. * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    norm = rho * N
    # Values in g_r array are also n_his
    g_r /= norm * V
    # print(r)
    # print(g_r)

    return r, g_r


# 1.1
def initGrid(l_domain, mass_density):
    """
    Function to determine particle positions and box size for a specified number of particles and density

    Start at the empty array coords in which we will place the coordinates of the particles.
    Create an additional variable called L which shall give the box dimension (float).
    Make sure to shift the coordinates to have (0,0,0) as the center of the box.

    :param l_domain: Length of box domain sides [Angstroms]
    :param mass_density: Mass density of particles [kg/m^3]
    """
    molecule_density = convertMassDensity(mass_density)
    nPart = int(np.ceil(molecule_density * l_domain ** 3))
    # print(nPart)

    # load empty array for coordinates
    molecules_coordinates = np.zeros((nPart, 3))
    # print(coords.shape[0])

    # Find number of particles in each lattice line -> round up to nearest in
    n = int(np.ceil(nPart ** (1 / 3)))
    # print(n)

    # define lattice spacing
    spac = l_domain / n

    # initiate lattice indexing
    index = np.zeros(3)
    # print(index)

    # assign particle positions
    for part in range(nPart):
        molecules_coordinates[part, :] = index * spac

        # advance particle position
        index[0] += 1

        # if last lattice point is reached jump to next line
        if index[0] == n:
            index[0] = 0
            index[1] += 1
            if index[1] == n:
                index[1] = 0
                index[2] += 1

    """
    plt.figure()
    plt.scatter(x, y)
    plt.show()
    """

    """
    r, g_r = rdf(coords, np.array([L, L, L]))
    plt.scatter(r, g_r, 1)
    # plt.title("")
    plt.ylabel("g(r)")
    plt.xlabel("r [A]")
    plt.savefig('rdf.png')
    plt.show()
    """

    """
    :Given code did not include "+ spac / 2" so there was an offset
    :return: numpy array containing coordinates centered around origin
    :return: box size
    """
    return molecules_coordinates + spac / 2 - l_domain / 2, nPart


# 1.2
def initVel(T, no_of_entities):
    """
    Function to initialize temperature

    :param T: System temperature [K]
    :param no_of_entities: Number of molecules in the system [-]
    :return: Vector of molecules velocities and scaling factor
    """
    # Generate random velocity
    # Given in notes -> As a rule of thumb, a fast moving atm should move at most O(1%) of its diameter in a timestep
    v_magnitude_max = (1 / 100) * sigma  # [Angstroms/fs]
    v_direction_max = np.sqrt((1 / 3) * (v_magnitude_max ** 2))

    v = np.zeros((no_of_entities, 3))
    for i in range(0, no_of_entities):
        v[i] = np.random.uniform(-v_direction_max, v_direction_max, size=3)
    v2 = np.sum(v ** 2, axis=1)
    # NOTE: We will not average across the 0.5 and mass pre-factors since they are taken as constants
    v2_average = np.average(v2)

    # System temperature with randomized velocities
    # T_init = (CH4_molecule_mass * v2_average * 1e10) / (ndim * co.k)
    T_init = (CH4_molar_mass * v2_average * 1e4) / (ndim * co.R * 1e-3)
    # print(T_init)

    scale_factor = T / T_init
    # print(scale_factor)
    v = np.sqrt(scale_factor) * v
    # print(v)

    return v


# 2.1 and 2.2
def LJ_forces(molecules_coordinates, l_domain, r_cut):
    """
    Function to calculate inter-molecular forces for each configuration

    :param molecules_coordinates: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: Forces [J/Angstrom] -> Equivalent of J/m which is N
    """
    no_of_entities = molecules_coordinates.shape[0]
    # print(no_of_entities)

    forces = np.zeros((no_of_entities, 3))
    np.set_printoptions(formatter={'float': lambda x: format(x, '1.5E')})

    for (i, coordinates_i) in enumerate(molecules_coordinates):
        # list containing Δx, Δy, Δz for each pair
        # CHECK THIS -> shouldn't be wrong if you offset this back to a left corner based domain
        # Shape -> (N, 3)
        d = (molecules_coordinates - molecules_coordinates[i] + l_domain / 2) % l_domain - l_domain / 2
        # print(d)
        # Add arbitrary large number so that r_ij_sq for the self-interaction goes beyond cut-off
        d[i] += 20
        # Radial distance between pairs
        # Shape -> (N)
        r_ij_sq = np.sum(d * d, axis=1)
        # Removes overlaps and replaces with infinitesimally small distance to result in large forces that repel away
        r_ij_sq = np.where(r_ij_sq == 0, 0.01, r_ij_sq)
        # print(r_ij_sq)

        # NOTE: WE SHALL NOT COMPUTE V_IJ BECAUSE IT WOULD BE COMPUTATIONALLY MORE EXPENSIVE TO PERFORM THE SQRT
        # OPERATION AND MATHEMATICALLY, IT IS UNNECESSARY. WHEN YOU OBTAIN THE FORMULA OF DU_DR, YOU WILL SEE THAT THE
        # DENOMINATORS IN THE BRACKETS ARE R^13 AND R^7, HOWEVER, THE UNIT VECTOR ACTUALLY CONTAINS ANOTHER R AS ITS
        # DENOMINATOR, SO IF WE MULTIPLY THIS IN, THE DENOMINATORS BECOME R^14 AND R^8. THE FACT THAT THE POWER OF THE
        # DENOMINATORS ARE IN MULTIPLES OF 2 ALLOWS US TO AVOID THE SQRT OPERATION.
        # v_ij = np.where(r_ij_sq <= r_cut ** 2, d / np.sqrt(r_ij_sq), np.zeros(3)) # v_ij -> Unit vector of r_ij

        # Filter out all r2 larger than rcut squared and get sigma^2/r^2 for all particles j>i. Necessary to set value
        # as 0 if rejected to maintain size of array
        # Shape -> (N, 3)
        sr2 = np.where(r_ij_sq <= r_cut ** 2, sigma_sq / r_ij_sq, 0)
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        # Force magnitude
        # Unit vectors are ji not ij so the signs are reversed which is why no -ve is in front
        dU_dr = ((24 * epslj) / r_ij_sq) * (2 * sr12 - sr6)
        # Vectorize dU_dr
        F = - d * dU_dr[:, np.newaxis]
        # Summation of all pair forces and storing the force vectors into a 2D array
        forces[i] = np.sum(F, axis=0)  # [J/Angstrom]

    # print(forces)

    return forces


# 3
def velocityVerlet(timestep, molecules_coordinates, l_domain, forces, v_old, r_cut):
    """
    Velocity-Verlet Algorithm function

    :param timestep: Time for each step [fs]
    :param molecules_coordinates: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param forces: Forces [N/Angstrom]
    :param v_old: Vector containing cartesian velocities  of each molecule [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: New configuration coordinates, new velocity vectors and force-field of new configuration
    """
    r_old = molecules_coordinates

    v_half_new = np.zeros((len(molecules_coordinates), 3))
    v_new = np.zeros((len(molecules_coordinates), 3))
    r_new = np.zeros((len(molecules_coordinates), 3))

    for i in range(0, len(molecules_coordinates)):
        r_new[i] = r_old[i] + (v_old[i] * timestep) + (forces[i] / CH4_molecule_mass) * 1e-7 * (timestep ** 2)
        # Bring back coordinates from ghost cells
        r_new[i] = np.where(r_new[i] > + l_domain / 2, r_new[i] - l_domain, r_new[i])
        r_new[i] = np.where(r_new[i] < - l_domain / 2, r_new[i] + l_domain, r_new[i])
        v_half_new[i] = v_old[i] + (forces[i] / (2 * CH4_molecule_mass)) * 1e-7 * timestep

    forces_new = LJ_forces(r_new, l_domain, r_cut)

    for i in range(0, len(molecules_coordinates)):
        v_new[i] = v_half_new[i] + (forces[i] / (2 * CH4_molecule_mass)) * 1e-7 * timestep
        # v_new[i] = np.round(v_new[i], 6)

    return r_new, v_new, forces_new


# 6 
def velocityVerletThermostat(timestep, T, Q, molecules_coordinates, l_domain, forces, v_old, r_cut):
    '''

    :param timestep: Time for each step [fs]
    :param T: System temperature [K]
    :param Q: Damping parameter [-]
    :param molecules_coordinates: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param forces: Forces [N/Angstrom]
    :param v_old: Vector containing cartesian velocities  of each molecule [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: New configuration coordinates, new velocity vectors and force-field of new configuration
    '''
    no_of_entities = molecules_coordinates.shape[0]

    r_old = molecules_coordinates
    zeta_old = np.zeros((len(molecules_coordinates), 3))

    zeta_half_new = np.zeros((len(molecules_coordinates), 3))
    v_half_new = np.zeros((len(molecules_coordinates), 3))

    r_new = np.zeros((len(molecules_coordinates), 3))
    zeta_new = np.zeros((len(molecules_coordinates), 3))
    v_new = np.zeros((len(molecules_coordinates), 3))

    U_kin = np.sum((0.5 * CH4_molecule_mass * (v_old ** 2))) / no_of_entities

    for i in range(0, len(molecules_coordinates)):
        # Calculate coordinates for new configuration r(t+delta t)
        r_new[i] = r_old[i] + (v_old[i] * timestep) + ((timestep ** 2) / 2) * (
                (forces[i] / CH4_molecule_mass) * 1e-7 - (zeta_old[i]) * v_old[i])
        # Bring back coordinates from ghost cells
        r_new[i] = np.where(r_new[i] >= + l_domain / 2, r_new[i] - l_domain, r_new[i])
        r_new[i] = np.where(r_new[i] <= - l_domain / 2, r_new[i] + l_domain, r_new[i])

        zeta_half_new[i] = zeta_old[i] + (timestep / (2 * Q)) * (U_kin - 1.5 * co.k * T)
        v_half_new[i] = v_old[i] + (timestep / 2) * (
                    (forces[i] / CH4_molecule_mass) * 1e-7 - zeta_half_new[i] * v_old[i])

    # Calculate forces for new configuration r(t+delta t)
    forces = LJ_forces(r_new, l_domain, r_cut)

    for i in range(0, len(molecules_coordinates)):
        zeta_new[i] = zeta_half_new[i] + (timestep / (2 * Q)) * (U_kin - 1.5 * co.k * T)
        v_new[i] = (v_half_new[i] + ((timestep / 2) * (forces[i] / CH4_molecule_mass)) * 1e10) / (
                1 + (timestep / 2) * zeta_new[i])

    return r_new, v_new, forces


# 4
def kineticEnergy(T, no_of_entities):
    """
    Function to calculate the kinetic energy of the system based on the temperature

    :param T: System temperature [K]
    :param no_of_entities: Number of molecules in the system [-]
    :return: Kinetic Energy [J]
    """
    U_kin = 0.5 * co.k * T * ndim * no_of_entities

    return U_kin


# 4
def potentialEnergy(molecules_coordinates, l_domain, r_cut=14):
    """
    Function to calculate the potential energy of the system using the truncated Lennard-Jones potential

    :param molecules_coordinates: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: Total Lennard-Jones potential of the system [J]
    """
    # molecules_coordinates = np.array(molecules_coordinates)
    no_of_entities = molecules_coordinates.shape[0]
    # Simulation box size, note, we shall work with angstroms [Angstroms]
    domain_volume = l_domain ** 3
    # Molecule density
    molecule_density = no_of_entities / domain_volume  # [1/Angstroms^3]

    # Initialized the Lennard-Jones parameters for the calculation of potential energy
    U_lj = 0

    sr3 = sigma_cu / (r_cut ** 3)

    for (i, coordinates_i) in enumerate(molecules_coordinates):
        # list containing Δx, Δy, Δz for each pair
        d = (molecules_coordinates[i + 1:] - coordinates_i + l_domain / 2) % l_domain - l_domain / 2
        r_ij_sq = np.sum(d * d, axis=1)
        # print(r_ij_sq)

        # filter out all r2 larger than rcut squared and get sigma^2/r^2 for all particles j>i
        sr2 = sigma_sq / r_ij_sq[r_ij_sq <= r_cut ** 2]
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        U_lj += np.sum(sr12 - sr6)

    U_lj = 4 * epslj * U_lj  # [J]
    # print(U_lj)
    U_ljtail = no_of_entities * (8 / 3) * co.pi * epslj * molecule_density * sigma_cu * (
            (1 / 3) * (sr3 * sr3 * sr3) - sr3)  # [J]
    # print(U_ljtail)
    U_pot = U_lj + U_ljtail  # [J]
    # print(U_pot)

    return U_pot


# 4
def temperature(particle_velocities):
    """
    Function to calculate system temperature

    :param particle_velocities: Vector containing cartesian velocities  of each molecule [Angstroms/fs]
    :return: System temperature [K]
    """
    v2 = particle_velocities ** 2
    average_Kinetic = 0.5 * CH4_molecule_mass * np.average(v2) * 1e-10

    T = average_Kinetic / (ndim * co.k)  # [K]
    # print(T)

    return T


# 4
def pressure(T, molecules_coordinates, l_domain, r_cut=14):
    """
    Function to calculate total system pressure

    :param T: System temperature [K]
    :param molecules_coordinates: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: System pressure [Pa]
    """
    # molecules_coordinates = np.array(molecules_coordinates)
    no_of_entities = molecules_coordinates.shape[0]
    # Simulation box size, note, we shall work with angstroms [Angstroms]
    domain_volume = l_domain ** 3
    # Molecule density
    molecule_density = no_of_entities / domain_volume  # [1/Angstroms^3]
    # Cutoff radius to prevent duplicate interactions [Angstroms] if condition implemented for Q3.1 later

    # Initialized the Lennard-Jones parameters for the calculation of pressure
    dU_dr = 0

    for (i, coordinates_i) in enumerate(molecules_coordinates):
        # list containing Δx, Δy, Δz for each pair
        d = (molecules_coordinates[i + 1:] - coordinates_i + l_domain / 2) % l_domain - l_domain / 2
        r_ij_sq = np.sum(d * d, axis=1)
        # print(r_ij_sq)

        # Filter out all r2 larger than rcut squared and get sigma^2/r^2 for all particles j>i
        sr2 = sigma_sq / r_ij_sq[r_ij_sq <= r_cut ** 2]
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        dU_dr += np.sum(sr6 - 2 * sr12)

    dU_dr = 24 * epslj * dU_dr
    P_tot = (molecule_density * k_B * T - (1 / (3 * domain_volume)) * dU_dr) * 1e30  # [Pa]
    # print(P_tot)

    return P_tot


def write_frame(coords, L, vels, forces, trajectory_name, step):
    '''
    function to write trajectory file in LAMMPS format

    In VMD you can visualize the motion of particles using this trajectory file.

    :param coords: coordinates
    :param vels: velocities
    :param forces: forces
    :param trajectory_name: trajectory filename

    :return:
    '''

    nPart = len(coords[:, 0])
    nDim = len(coords[0, :])
    with open(trajectory_name, 'a') as file:
        file.write('ITEM: TIMESTEP\n')
        file.write('%i\n' % step)
        file.write('ITEM: NUMBER OF ATOMS\n')
        file.write('%i\n' % nPart)
        file.write('ITEM: BOX BOUNDS pp pp pp\n')
        for dim in range(nDim):
            file.write('%.6f %.6f\n' % (-0.5 * L, 0.5 * L))
        for dim in range(3 - nDim):
            file.write('%.6f %.6f\n' % (0, 0))
        file.write('ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz\n')

        temp = np.zeros((nPart, 9))
        for dim in range(nDim):
            temp[:, dim] = coords[:, dim]
            temp[:, dim + 3] = vels[:, dim]
            temp[:, dim + 6] = forces[:, dim]

        for part in range(nPart):
            file.write('%i %i %.4f %.4f %.4f %.6f %.6f %.6f %.4f %.4f %.4f\n' % (part + 1, 1, *temp[part, :]))


def read_lammps_trj(lammps_trj_file):
    def read_lammps_frame(trj):
        """Load a frame from a LAMMPS dump file.

        Args:
            trj (file): LAMMPS dump file of format 'ID type x y z' or
                                                   'ID type x y z vx vy vz' or
                                                   'ID type x y z fz'
            read_velocities (bool): if True, reads velocity data from file
            read_zforces (bool): if True, reads zforces data from file

        Returns:
            xyz (numpy.ndarray):
            types (numpy.ndarray):
            step (int):
            box (groupy Box object):
            vxyz (numpy.ndarray):
            fz (numpy.ndarray):
        """
        # --- begin header ---
        trj.readline()  # text "ITEM: TIMESTEP"
        step = int(trj.readline())  # timestep
        trj.readline()  # text "ITEM: NUMBER OF ATOMS"
        n_atoms = int(trj.readline())  # num atoms
        trj.readline()  # text "ITEM: BOX BOUNDS pp pp pp"
        Lx = trj.readline().split()  # x-dim of box
        Ly = trj.readline().split()  # y-dim of box
        Lz = trj.readline().split()  # z-dim of box
        L = np.array([float(Lx[1]) - float(Lx[0]),
                      float(Ly[1]) - float(Ly[0]),
                      float(Lz[1]) - float(Lz[0])])
        trj.readline()  # text
        # --- end header ---

        xyz = np.empty(shape=(n_atoms, 3))
        xyz[:] = np.nan
        types = np.empty(shape=(n_atoms), dtype='int')
        vxyz = np.empty(shape=(n_atoms, 3))
        vxyz[:] = np.nan
        fxyz = np.empty(shape=(n_atoms, 3))
        fxyz[:] = np.nan

        # --- begin body ---

        IDs = []
        for i in range(n_atoms):
            temp = trj.readline().split()
            a_ID = int(temp[0]) - 0  # atom ID
            xyz[a_ID - 1] = [float(x) for x in temp[2:5]]  # coordinates
            types[a_ID - 1] = int(temp[1])  # atom type
            vxyz[a_ID - 1] = [float(x) for x in temp[5:8]]  # velocities
            fxyz[a_ID - 1] = [float(x) for x in temp[8:11]]  # map(float, temp[5]) # z-forces

        # --- end body ---
        return xyz, types, step, L, vxyz, fxyz

    xyz = {}
    vel = {}
    forces = {}
    with open(lammps_trj_file, 'r') as f:
        READING = True
        c = 0
        while READING:
            try:
                xyz[c], _, _, _, vel[c], forces[c] = read_lammps_frame(f)
                c += 1
            except:
                READING = False

    return xyz, vel, forces


# 5
def MD_CYCLE(simulation_time, timestep, L, coordinates, force, velocity, r_cut):
    """
    Function to run an MD loop for the specified duration and timestep.

    :param simulation_time: Total simulation time [fs]
    :param timestep: Time for each step [fs]
    :param L: Length of box domain sides [Angstroms]
    :param coordinates:
    :param force:
    :param velocity:
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return:
    """
    steps = simulation_time / timestep
    sample_frequency = 100
    sample_interval = steps / sample_frequency

    r_old = coordinates
    force_old = force
    v_old = velocity

    for i in range(0, simulation_time):
        r_new, v_new, force_new = velocityVerlet(timestep, r_old, L, force_old, v_old, r_cut)
        print(i)

        if (i + 1) % sample_interval == 0:
            write_frame(r_new, L, v_new, force_new, 'Declan_trj.lammps', (i + 1) * timestep)
            # NOT WRITING FORCE FOR SOME REASON BUT IF WE PRINT FORCE, THERE ARE NUMBERS

        r_old = r_new
        v_old = v_new
        force_old = force_new

"""coordinates_1, l_domain_1 = initGrid(30, 0.5 * 358.4)
coordinates_2, l_domain_2 = initGrid(30,358.4)
coordinates_3, l_domain_3 = initGrid(30,2 * 358.4)

r, g_r1 = rdf(coordinates_1, np.array([l_domain_1,l_domain_1,l_domain_1]))
g_r2 = rdf(coordinates_2, np.array([l_domain_2,l_domain_2,l_domain_2]))[1]
g_r3 = rdf(coordinates_3, np.array([l_domain_3,l_domain_3,l_domain_3]))[1]
#plt.scatter(r, g_r, 3)
plt.plot(r, g_r1, '-o', markersize = 3)
plt.plot(r, g_r2, '-o', markersize = 3)
# plt.plot(r, g_r3, '-o', markersize = 3)
# plt.title("")
plt.ylabel("g(r)")
plt.xlabel("r [A]")
plt.savefig('rdf.png')
plt.show()"""

# Initializing domain
l_domain = 30
coord_array, no_of_molecules = initGrid(l_domain, 358.4)
# Initializing velocity
v_array = initVel(150, no_of_molecules)
# Initializing force
force_array = LJ_forces(coord_array, l_domain, 14)

MD_CYCLE(3000, 1, l_domain, coord_array, force_array, v_array, 14)
xyz, vel, F = read_lammps_trj('Declan_trj.lammps')
# xyz, vel, forces = read_lammps_trj('trj.lammps')
print(xyz)
print(vel)
print(F)

