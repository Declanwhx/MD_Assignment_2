import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.constants as co
import re

######################### CONSTANTS #########################
# Boltzmann Constant
k_B = co.k  # [J/K]
# Lennard-Jones variables -> Depth of well and collisional diameter
epslj = 148 * co.k  # [J]
sigma = 3.73  # [Angstroms]
sigma_sq = sigma ** 2  # [Angstroms^2]
sigma_cu = sigma ** 3  # [Angstroms^3]
CH4_molar_mass = 16.04 * (10 ** -3) # [kg/mol]
CH4_molecule_mass = CH4_molar_mass / co.N_A # [kg]
# Degrees of freedom
dof = 3  # [-]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # FUNCTIONS # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def convertMassDensity(mass_density):
    """
    Function that takes in the mass density and converts it to molecule density

    :param mass_density: Mass density of system [kg/m^3]
    :return: Molecule density [1/Angstrom^3]
    """
    molecule_density = ((mass_density / CH4_molar_mass) * co.N_A) / (10 ** 30)
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
    return molecules_coordinates + spac / 2 - l_domain / 2, l_domain


# 1.2 CURRENTLY I AM RELOOKING AT THIS.
def initVel(T, no_of_entities):
    """
    Function to initialize temperature

    :param T: System temperature [K]
    :param no_of_entities: Number of molecules in the system [-]
    :return: Vector of molecules velocities
    """
    # Generate random velocity
    # Given in notes -> As a rule of thumb, a fast moving atm should move at most O(1%) of its diameter in a timestep
    v_magnitude_max = (1/100) * sigma # [Angstroms/fs]
    v_direction_max = np.sqrt((1 / 3) * (v_magnitude_max ** 2))

    v = np.zeros((no_of_entities, 3))
    for i in range(0, no_of_entities):
        v[i] = np.random.uniform(-v_direction_max, v_direction_max, size=3)
    
    #print(v)
    v2 = np.sum(v ** 2, axis = 1 )
    # print(v2)
    v2_average = np.average(v2)
    # print(v2_average)

    # System temperature with randomized velocities
    # CHECK UNITS HERE -> should be correct now but just in case
    T_init = (CH4_molecule_mass * v2_average * (1e-10)) / (dof * co.k)
    #print(T_init)

    scale_factor = T / T_init
    #print(scale_factor)
    v = np.sqrt(scale_factor) * v
    # print(v)

    return v


# 2.1 and 2.2
def LJ_forces(molecules_coordinates, l_domain, r_cut=14):
    """
    Function to calculate inter-molecular forces for each configuration

    :param molecules_coordinates: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: Forces [N/Angstrom] -> Equivalent of J/m which is N
    """
    no_of_entities = molecules_coordinates.shape[0]
    # print(no_of_entities)

    forces = np.zeros((no_of_entities, 3))

    for (i, coordinates_i) in enumerate(molecules_coordinates):
        # list containing Δx, Δy, Δz for each pair
        # CHECK THIS -> shouldn't be wrong if you offset this back to a left corner based domain
        # Shape -> (N, 3)
        d = (molecules_coordinates - coordinates_i + l_domain / 2) % l_domain - l_domain / 2
        # d = (molecules_coordinates[i + 1:] - coordinates_i + l_domain / 2) % l_domain - l_domain / 2
        # print(d)
        # Radial distance between pairs
        # Shape -> (N)
        r_ij_sq = np.sum(d * d, axis=1)
        r_ij_sq[i] = r_cut ** 2
        # Removes overlaps and replaces with infinitesimally small distance to result in large forces that repel away
        r_ij_sq = np.where(r_ij_sq == 0, 0.00001, r_ij_sq)
        # print(r_ij_sq)

        # NOTE: WE SHALL NOT COMPUTE V_IJ BECAUSE IT WOULD BE COMPUTATIONALLY MORE EXPENSIVE TO PERFORM THE SQRT
        # OPERATION AND MATHEMATICALLY, IT IS UNNECESSARY. WHEN YOU OBTAIN THE FORMULA OF DU_DR, YOU WILL SEE THAT THE
        # DENOMINATORS IN THE BRACKETS ARE R^13 AND R^7, HOWEVER, THE UNIT VECTOR ACTUALLY CONTAINS ANOTHER R AS ITS
        # DENOMINATOR, SO IF WE MULTIPLY THIS IN, THE DENOMINATORS BECOME R^14 AND R^8. THE FACT THAT THE POWER OF THE
        # DENOMINATORS ARE IN MULTIPLES OF 2 ALLOWS US TO AVOID THE SQRT OPERATION.
        # v_ij = np.where(r_ij_sq <= r_cut ** 2, d / np.sqrt(r_ij_sq), np.zeros(3)) # v_ij -> Unit vector of r_ij

        # Filter out all r2 larger than rcut squared and get sigma^2/r^2 for all particles j>i. Necessary to set value
        # as 0 if rejected to maintain size of array
        sr2 = np.where(r_ij_sq <= r_cut ** 2, sigma_sq / r_ij_sq, 0)
        sr6 = (sr2 ** 3) / r_ij_sq
        sr12 = (sr6 ** 2) / r_ij_sq

        # Force magnitude
        dU_dr = ((24 * epslj) / r_ij_sq) * (2 * sr12 - sr6)
        # Vectorize dU_dr
        F = - d * dU_dr[:, np.newaxis]
        # print(dU_dr)
        # Summation of all pair forces and storing the force vectors into a 2D array
        forces[i] = np.sum(F, axis=0)  # [J/Angstrom]
        # print(forces[i])

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
        r_new[i] = r_old[i] + (v_old[i] * timestep) + (forces[i] / CH4_molecule_mass) * (1e2) * (timestep ** 2)
        # Bring back coordinates from ghost cells
        r_new[i] = np.where(r_new[i] > + l_domain / 2, r_new[i] - l_domain, r_new[i])
        r_new[i] = np.where(r_new[i] < - l_domain / 2, r_new[i] + l_domain, r_new[i])
        v_half_new[i] = v_old[i] + (forces[i] / (2 * CH4_molecule_mass)) * (1e2) * timestep

    forces = LJ_forces(r_new, l_domain, r_cut)

    for i in range(0, len(molecules_coordinates)):
        v_new[i] = v_half_new[i] + (forces[i] / (2 * CH4_molecule_mass)) * (1e2) * timestep
        v_new[i] = np.round(v_new[i], 6)

    return r_new, v_new, forces


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

    U_kin = np.sum((0.5 * CH4_molecule_mass * ((v_old) ** 2))) / no_of_entities

    for i in range(0, len(molecules_coordinates)):
        # Calculate coordinates for new configuration r(t+delta t)
        r_new[i] = r_old[i] + (v_old[i] * timestep) + ((timestep ** 2) / 2) * (
                    (forces[i] / CH4_molecule_mass)  * (1e2) - (zeta_old[i]) * v_old[i])
        # Bring back coordinates from ghost cells
        r_new[i] = np.where(r_new[i] >= + l_domain / 2, r_new[i] - l_domain, r_new[i])
        r_new[i] = np.where(r_new[i] <= - l_domain / 2, r_new[i] + l_domain, r_new[i])

        zeta_half_new[i] = zeta_old[i] + (timestep / (2 * Q)) * (U_kin - 1.5 * co.k * T)
        v_half_new[i] = v_old[i] + (timestep / 2) * ((forces[i] / CH4_molecule_mass)  * (1e2) - zeta_half_new[i] * v_old[i])

    # Calculate forces for new configuration r(t+delta t)
    forces = LJ_forces(r_new, l_domain, r_cut)

    for i in range(0, len(molecules_coordinates)):
        zeta_new[i] = zeta_half_new[i] + (timestep / (2 * Q)) * (U_kin - 1.5 * co.k * T)
        v_new[i] = (v_half_new[i] + ((timestep / 2) * (forces[i] / CH4_molecule_mass)) * (1e2))/ (
                1 + (timestep / 2) * zeta_new[i])

    return r_new, v_new, forces


# 4
def kineticEnergy(T, molecules_coordinates):
    """
    Function to calculate the kinetic energy of the system based on the temperature

    :param T: System temperature [K]
    :param molecules_coordinates: Array of molecules coordinates -> Coordinates not important here, only array shape
    :return: Kinetic Energy [J]
    """
    U_kin = 0.5 * co.k * T * dof * molecules_coordinates.shape[0]

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

    :param particle_velocities: Vector containing cartesian velocities  of each molecule [Angstroms]
    :return: System temperature [K]
    """
    v2 = particle_velocities ** 2
    #CHECK UNITS -> THIS NEEDS CHANGING(not for you Jelle, this is just for me to track)
    average_Kinetic = 0.5 * CH4_molecule_mass * np.average(v2)

    T = average_Kinetic / (dof * co.k) # [K]
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
    #CHECK UNITS (not for you Jelle, this is just for me to track)
    P_tot = (molecule_density * k_B * T - (1 / (3 * domain_volume)) * dU_dr) * 1e30  # [Pa]
    # print(P_tot)

    return P_tot


def write_frame(molecules_coordinates, l_domain, v, forces, trajectory_name, timestep):
    """

    :param molecules_coordinates: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param v:
    :param forces:
    :param trajectory_name:
    :param timestep: Time for each step [fs]
    :return:
    """

    nPart = len(molecules_coordinates[:, 0])
    nDim = len(molecules_coordinates[0, :])
    with open(trajectory_name, 'a') as file:
        file.write('ITEM: TIMESTEP\n')
        file.write('%i\n' % timestep)
        file.write('ITEM: NUMBER OF ATOMS\n')
        file.write('%i\n' % nPart)
        file.write('ITEM: BOX BOUNDS pp pp pp\n')
        for dim in range(nDim):
            file.write('%.6f %.6f\n' % (-0.5 * l_domain[dim], 0.5 * l_domain[dim]))
        for dim in range(3 - nDim):
            file.write('%.6f %.6f\n' % (0, 0))
        file.write('ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz\n')

        temp = np.zeros((nPart, 9))
        for dim in range(nDim):
            temp[:, dim] = molecules_coordinates[:, dim]
            temp[:, dim + 3] = v[:, dim]
            temp[:, dim + 6] = forces[:, dim]

        for part in range(nPart):
            file.write('%i %i %.4f %.4f %.4f %.6f %.6f %.6f %.4f %.4f %.4f\n' % (part + 1, 1, *temp[part, :]))


def read_lammps_data(data_file, verbose=False):
    """Reads a LAMMPS data file
        Atoms
        Velocities
    Returns:
        lmp_data (dict):
            'xyz': xyz (numpy.ndarray)
            'vel': vxyz (numpy.ndarray)
        box (numpy.ndarray): box dimensions
    """
    print("Reading '" + data_file + "'")
    with open(data_file, 'r') as f:
        data_lines = f.readlines()

    # TODO: improve robustness of xlo regex
    directives = re.compile(r"""
        ((?P<n_atoms>\s*\d+\s+atoms)
        |
        (?P<box>.+xlo)
        |
        (?P<Atoms>\s*Atoms)
        |
        (?P<Velocities>\s*Velocities))
        """, re.VERBOSE)

    i = 0
    while i < len(data_lines):
        match = directives.match(data_lines[i])
        if match:
            if verbose:
                print(match.groups())

            elif match.group('n_atoms'):
                fields = data_lines.pop(i).split()
                n_atoms = int(fields[0])
                xyz = np.empty(shape=(n_atoms, 3))
                vxyz = np.empty(shape=(n_atoms, 3))

            elif match.group('box'):
                dims = np.zeros(shape=(3, 2))
                for j in range(3):
                    fields = [float(x) for x in data_lines.pop(i).split()[:2]]
                    dims[j, 0] = fields[0]
                    dims[j, 1] = fields[1]
                L = dims[:, 1] - dims[:, 0]

            elif match.group('Atoms'):
                if verbose:
                    print('Parsing Atoms...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    a_id = int(fields[0])
                    xyz[a_id - 1] = np.array([float(fields[2]),
                                              float(fields[3]),
                                              float(fields[4])])

            elif match.group('Velocities'):
                if verbose:
                    print('Parsing Velocities...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    va_id = int(fields[0])
                    vxyz[va_id - 1] = np.array([float(fields[1]),
                                                float(fields[2]),
                                                float(fields[3])])

            else:
                i += 1
        else:
            i += 1

    return xyz, vxyz, L


# 5
def MD_CYCLE(simulation_time, timestep, T, mass_density, l_domain, r_cut):
    """
    Function to run an MD loop for the specified duration and timestep.

    :param simulation_time: Total simulation time [fs]
    :param timestep: Time for each step [fs]
    :param T: System temperature [K]
    :param mass_density: Mass density of system [kg/m^3]
    :param l_domain: Length of box domain sides [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return:
    """
    steps = simulation_time / timestep
    sample_frequency = 100
    sample_interval = steps / sample_frequency

    # Initializing domain
    r_old, L = initGrid(l_domain, mass_density)
    # Initializing velocity
    v_old = initVel(T, r_old.shape[0])
    # Initializing force
    force_old = LJ_forces(r_old, l_domain, r_cut)

    for i in range(0, simulation_time):
        r_new, v_new, force_new = velocityVerlet(timestep, r_old, l_domain, force_old, v_old, r_cut)

        # print(coordinates_old)
        if (i + 1) % sample_interval == 0:
            write_frame(r_new, l_domain, v_new, force_new, 'Declan_trj.lammps',
                        (i + 1) * timestep)

        r_old = r_new
        v_old = v_new
        force_old = force_new

    return



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

MD_CYCLE(3000, 1, 150, 358.4, 30, 14)
"""xyz, vel, L = read_lammps_data('lammps.data')
print(xyz)
print(vel)
print(L)"""
