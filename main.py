import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.constants as co
import lammps_logfile

# # # # # # # # # # # # # # # CONSTANTS # # # # # # # # # # # # # # #
k_B = co.k  # Boltzmann Constant [J/K]
epslj = 148 * co.k  # Depth of well [J]
sigma = 3.73  # Collisional diameter [Angstroms]
sigma_sq = sigma ** 2  # [Angstroms^2]
sigma_cu = sigma ** 3  # [Angstroms^3]
CH4_molar_mass = 16.04  # CH4 Molar mass [g/mol]
CH4_molecule_mass = CH4_molar_mass * (1 / co.N_A)  # Mass of a single CH4 molecule [g]
ndim = 3  # Degrees of freedom [-]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # FUNCTIONS # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def convertMassDensity(mass_density):
    """
    Function that takes in the mass density and converts it to molecule density

    :param mass_density: Mass density of system [kg/m^3]
    :return: Molecule density [1/Angstrom^3]
    """
    molecule_density = ((mass_density / (CH4_molar_mass * 1e-3)) * co.N_A) / (1e30)
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
        # PBC
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
    no_of_entities = int(np.ceil(molecule_density * l_domain ** 3))
    # print(no_of_entities)

    # load empty array for coordinates
    coordinates_array = np.zeros((no_of_entities, 3))
    # print(coords.shape[0])

    # Find number of particles in each lattice line -> round up to nearest in
    n = int(np.ceil(no_of_entities ** (1 / 3)))
    # print(n)

    # define lattice spacing
    spac = l_domain / n

    # initiate lattice indexing
    index = np.zeros(3)
    # print(index)

    # assign particle positions
    for part in range(no_of_entities):
        coordinates_array[part, :] = np.round(index * spac, 2)

        # advance particle position
        index[0] += 1

        # if last lattice point is reached jump to next line
        if index[0] == n:
            index[0] = 0
            index[1] += 1
            if index[1] == n:
                index[1] = 0
                index[2] += 1

    # Uncomment this block to generate .xyz files for visualization on VMD
    """
    molecules_coordinates = open('generated_box_3.xyz', "w")
    molecules_coordinates.write(str(no_of_entities) + "\n")
    molecules_coordinates.write("box.pdb \n")
    for i in range(0, no_of_entities):
        molecules_coordinates.write(
            "{0:<6}{1:>12}{2:>15}{3:>16}".format("C", coordinates_array[i][0], coordinates_array[i][1],
                                                 coordinates_array[i][2]) + "\n")
    """

    """
    :Given code did not include "+ spac / 2" so there was an offset
    :return: numpy array containing coordinates centered around origin
    :return: box size
    """

    return coordinates_array + spac / 2 - l_domain / 2, no_of_entities


# 1.2
def initVel(T, no_of_entities):
    """
    Function to initialize temperature

    :param T: System temperature [K]
    :param no_of_entities: Number of molecules in the system [-]
    :return: Vector of molecules velocities [Angstrom/fs]
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
    T_init = (CH4_molar_mass * v2_average * 1e4) / (ndim * co.R * 1e-3)
    # print(T_init)

    scale_factor = T / T_init
    # print(scale_factor)
    v = np.sqrt(scale_factor) * v
    # print(v)

    return v


# 2.1 and 2.2
def LJ_forces(coordinates_array, l_domain, r_cut):
    """
    Function to calculate inter-molecular forces for each configuration

    :param coordinates_array: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: Forces [J/Angstrom] -> Equivalent of J/m which is N
    """
    no_of_entities = coordinates_array.shape[0]

    forces = np.zeros((no_of_entities, 3))

    for (i, coordinates_i) in enumerate(coordinates_array):
        # list containing Δx, Δy, Δz for each pair. Shape -> (N, 3)
        d = (coordinates_array - coordinates_array[i] + l_domain / 2) % l_domain - l_domain / 2
        # Radial distance between pairs. Shape -> (N)
        r_ij_sq = np.sum(d * d, axis=1)
        # Removing self-interaction artefact
        r_ij_sq[i] = r_cut
        # Removes overlaps and replaces with infinitesimally small distance to result in large forces that repel away
        r_ij_sq = np.where(r_ij_sq == 0, 0.001, r_ij_sq)
        # print(r_ij_sq)

        # NOTE: WE SHALL NOT COMPUTE V_IJ (R_IJ IN THIS CODE) BECAUSE IT WOULD BE COMPUTATIONALLY MORE EXPENSIVE TO
        # PERFORM THE SQRT OPERATION AND MATHEMATICALLY, IT IS UNNECESSARY. WHEN YOU OBTAIN THE FORMULA OF DU_DR,
        # YOU WILL SEE THAT THE DENOMINATORS IN THE BRACKETS ARE R^13 AND R^7, HOWEVER, THE UNIT VECTOR ACTUALLY
        # CONTAINS ANOTHER R AS ITS DENOMINATOR, SO IF WE MULTIPLY THIS IN, THE DENOMINATORS BECOME R^14 AND R^8. THE
        # FACT THAT THE POWER OF THE DENOMINATORS ARE IN MULTIPLES OF 2 ALLOWS US TO AVOID THE SQRT OPERATION.
        # v_ij = np.where(r_ij_sq <= r_cut ** 2, d / np.sqrt(r_ij_sq), np.zeros(3)) # v_ij -> Unit vector of r_ij

        # Filter out all r2 larger than rcut squared and get sigma^2/r^2 for all pairs. Necessary to set value
        # as 0 if rejected to maintain size of array. Shape -> (N, 3)
        sr2 = np.where(r_ij_sq <= r_cut ** 2, sigma_sq / r_ij_sq, 0)
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        # Force magnitude
        # NOTE: UNIT VECTORS ARE JI INSTEAD OF IJ, SO THE SIGNS ARE REVERSED, WHICH IS WHY NO NEGATIVE SIGN IS IN FRONT.
        dU_dr = ((24 * epslj) / r_ij_sq) * (2 * sr12 - sr6)
        # Vectorize dU_dr
        F = - d * dU_dr[:, np.newaxis]
        # Summation of all pair forces and storing the force vectors into a 2D array
        forces[i] = np.sum(F, axis=0)  # [J/Angstrom]

    # print(forces)

    return forces


# 4
# Q3 below, kineticEnergy() was used for velocityVerletThermostat() so I placed the state variable functions before that
def kineticEnergy(velocity_field):
    """
    Function to calculate the instantaneous kinetic energy of the system

    :param velocity_field: Array containing [u, v, w] velocities of each molecule -> Array shape = (nParts, 3)
    :return: Kinetic Energy [J]
    """
    v2 = np.sum(velocity_field ** 2, axis=1)
    U_kin = 0.5 * CH4_molar_mass * np.sum(v2) * 1e4  # [kJ/mol]

    # print(U_kin)
    return U_kin


# 4
def potentialEnergy(coordinates_array, l_domain, r_cut=14):
    """
    Function to calculate the instantaneous potential energy of the system using the truncated Lennard-Jones potential

    :param coordinates_array: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: Total Lennard-Jones potential of the system [J]
    """
    # coordinates_array = np.array(coordinates_array)
    no_of_entities = coordinates_array.shape[0]
    # Simulation box size, note, we shall work with angstroms [Angstroms]
    domain_volume = l_domain ** 3
    # Molecule density
    molecule_density = no_of_entities / domain_volume  # [1/Angstroms^3]

    # Initialized the Lennard-Jones parameters for the calculation of potential energy
    U_lj = 0

    for (i, coordinates_i) in enumerate(coordinates_array):
        # list containing Δx, Δy, Δz for each pair
        d = (coordinates_array[i + 1:] - coordinates_i + l_domain / 2) % l_domain - l_domain / 2
        r_ij_sq = np.sum(d * d, axis=1)
        # print(r_ij_sq)
        r_ij_sq = np.where(r_ij_sq == 0, 0.001, r_ij_sq)

        # filter out all r2 larger than rcut squared and get sigma^2/r^2 for all particles j>i
        sr2 = sigma_sq / r_ij_sq[r_ij_sq <= r_cut ** 2]
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        U_lj += np.sum(sr12 - sr6)

    U_lj = 4 * ((epslj / co.k) * co.R * 1e-3) * U_lj
    U_pot = U_lj  # [kJ/mol]
    # print(U_pot)

    return U_pot


# 4
def temperature(velocity_field):
    """
    Function to calculate system temperature

    :param velocity_field: Vector containing cartesian velocities  of each molecule [Angstroms/fs]
    :return: System temperature [K]
    """
    v2 = np.sum(velocity_field ** 2, axis=1)
    KE = 0.5 * CH4_molar_mass * v2 * 1e4
    KE_average = np.average(KE)

    # T = average_kinetic / (ndim * co.k)
    T = KE_average / (ndim / 2 * co.R * 1e-3)  # [K]
    # print(T)

    return T


# 4
def pressure(T, coordinates_array, l_domain, r_cut=14):
    """
    Function to calculate total system pressure

    :param T: System temperature [K]
    :param coordinates_array: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: Total system pressure [Pa]
    """
    no_of_entities = coordinates_array.shape[0]
    # Simulation box size [Angstroms^3]
    domain_volume = l_domain ** 3
    # Molecule density
    molecule_density = no_of_entities / domain_volume  # [1/Angstroms^3]

    # Initialized the Lennard-Jones parameters for the calculation of pressure
    dU_dr = 0

    for (i, coordinates_i) in enumerate(coordinates_array):
        # list containing Δx, Δy, Δz for each pair
        d = (coordinates_array[i + 1:] - coordinates_i + l_domain / 2) % l_domain - l_domain / 2
        r_ij_sq = np.sum(d * d, axis=1)
        # print(r_ij_sq)
        r_ij_sq = np.where(r_ij_sq == 0, 0.001, r_ij_sq)

        # Filter out all r2 larger than rcut squared and get sigma^2/r^2 for all particles j>i
        sr2 = sigma_sq / r_ij_sq[r_ij_sq <= r_cut ** 2]
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2

        dU_dr += np.sum(sr6 - 2 * sr12)

    dU_dr = 24 * epslj * dU_dr
    P_tot = (molecule_density * k_B * T - (1 / (3 * domain_volume)) * dU_dr) * 1e30 / co.atm  # [atm]
    # print(P_tot)

    return P_tot


# 3
def velocityVerlet(timestep, coordinates_array, l_domain, forces, v_old, r_cut):
    """
    Function that executes one cycle of the MD-NVE cycle using the Velocity-Verlet Algorithm function

    :param timestep: Time for each step [fs]
    :param coordinates_array: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param forces: Forces [N/Angstrom]
    :param v_old: Vector containing cartesian velocities  of each molecule [Angstroms]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: New configuration coordinates, new velocity vectors and force-field of new configuration
    """
    r_old = coordinates_array

    r_new = r_old + (v_old * timestep) + (forces / CH4_molecule_mass) * 1e-7 * (timestep ** 2)
    # Bring back coordinates from ghost cells
    r_new = np.where(r_new > + l_domain / 2, r_new - l_domain, r_new)
    r_new = np.where(r_new < - l_domain / 2, r_new + l_domain, r_new)
    v_half_new = v_old + (forces / (2 * CH4_molecule_mass)) * 1e-7 * timestep

    forces_new = LJ_forces(r_new, l_domain, r_cut)

    v_new = v_half_new + (forces / (2 * CH4_molecule_mass)) * 1e-7 * timestep

    return r_new, v_new, forces_new


# 6
def velocityVerletThermostat(timestep, T, Q, coordinates_array, l_domain, forces, v_old, zeta_old, r_cut):
    """
    Function that executes one cycle of the MD-NVT cycle using the Nose-Hoover thermostat

    :param timestep: Time for each step [fs]
    :param T: System temperature [K]
    :param Q: Damping parameter [-]
    :param coordinates_array: Array containing cartesian coordinates of each molecule [Angstroms]
    :param l_domain: Length of box domain sides [Angstroms]
    :param forces: Forces [N/Angstrom]
    :param v_old: Vector containing cartesian velocities  of each molecule [Angstroms]
    :param zeta_old: 'Friction' integral for the calibration of temperature
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: New configuration coordinates, new velocity vectors, force-field of new configuration and 'Friction'
    integral array
    """
    N = coordinates_array.shape[0]
    # print(N)

    r_old = coordinates_array

    U_kin = kineticEnergy(v_old)
    # 1E-7 for conversion to Angstrom/fs^2 and to convert kilo.
    r_new = r_old + (v_old * timestep) + ((forces / CH4_molecule_mass) * 1e-7 - (zeta_old * v_old)) * (
                (timestep ** 2) / 2)
    r_new = np.where(r_new > + l_domain / 2, r_new - l_domain, r_new)
    r_new = np.where(r_new < - l_domain / 2, r_new + l_domain, r_new)

    # U_kin here is in kJ/mol so 3/2kbT has been multiplied by N_A and 1E-3 to maintain unit consistency
    zeta_half_new = zeta_old + ((U_kin / N) - ((3 / 2) * co.R * T) * 1e-3) * (timestep / (2 * Q))
    v_half_new = v_old + ((forces / CH4_molecule_mass) * 1e-7 - (zeta_half_new * v_old)) * (timestep / 2)

    forces = LJ_forces(r_new, l_domain, r_cut)

    zeta_new = zeta_half_new + (timestep / (2 * Q)) * ((U_kin / N) - ((3 / 2) * co.R * T) * 1e-3)
    v_new = (v_half_new + ((timestep / 2) * (forces / CH4_molecule_mass) * 1e-7)) / (1 + (timestep / 2) * zeta_new)

    T_new = temperature(v_new)

    return r_new, v_new, forces, zeta_new, T_new


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


def postProcStateVar(file='log.lammps'):
    log = lammps_logfile.File(file)

    time = log.get("Step")
    temperature = log.get("Temp")
    # Units conversion needed
    pressure = log.get("Press")
    U_kin = log.get("KinEng")
    U_kin *= 4.184
    U_Pot = log.get("PotEng")
    U_Pot *= 4.184

    return temperature, pressure, U_kin, U_Pot


# 5
def MD_NVE(simulation_time, timestep, L, coordinates, force, velocity, r_cut):
    """
    Function to run an MD-NVE loop for the specified duration and timestep.

    :param simulation_time: Total simulation time [fs]
    :param timestep: Time for each step [fs]
    :param L: Length of box domain sides [Angstroms]
    :param coordinates: Array containing the coordinates of molecules at a certain time [Angstrom]
    :param force: Array containing the force field of molecules at a certain time [J/Angstrom]
    :param velocity: Array containing the velocities of molecules at a certain time [Angstrom/fs]
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: State variables kinetic energy, potential energy, temperature and pressure
    """
    start = time.time()

    trajectory_filename = 'trajectory_1.lammps'

    steps = int(simulation_time / timestep)
    sample_frequency = 100
    sample_interval = int(steps / sample_frequency)

    r_old = coordinates
    force_old = force
    v_old = velocity

    U_kin = np.zeros(sample_frequency + 1)
    U_lj = np.zeros(sample_frequency + 1)
    T = np.zeros(sample_frequency + 1)
    p = np.zeros(sample_frequency + 1)
    time_array = np.zeros(sample_frequency + 1)

    U_kin[0] = kineticEnergy(v_old)
    U_lj[0] = potentialEnergy(r_old, L, r_cut)
    T[0] = temperature(v_old)
    p[0] = pressure(T[0], r_old, L, r_cut)

    counter = 1

    for i in range(1, simulation_time + 1):
        r_new, v_new, force_new = velocityVerlet(timestep, r_old, L, force_old, v_old, r_cut)
        # print(i)

        if i % sample_interval == 0:
            print(i, 'Sampling')
            U_kin[counter] = kineticEnergy(v_new)
            U_lj[counter] = potentialEnergy(r_new, L, r_cut)
            T[counter] = temperature(v_new)
            p[counter] = pressure(T[counter], r_new, L, r_cut)
            time_array[counter] = i
            counter += 1

            # print(T)
            write_frame(r_new, L, v_new, force_new, trajectory_filename, (i + 1) * timestep)

        r_old = r_new
        v_old = v_new
        force_old = force_new

    end = time.time()
    print("The time of execution of above program is :", (end - start) / 60, "mins")
    lammps_temperature, lammps_pressure, lammps_kin, lammps_pot = postProcStateVar(
        './DECLAN_VERIFY1/GIVEN_DATA/log.lammps')

    plt.plot(time_array, U_kin, '-o', markersize=0.5)
    plt.plot(time_array, lammps_kin, '-o', markersize=0.5)
    plt.ylabel("Kinetic Energy [kJ/mol]")
    plt.xlabel("Time [fs]")
    plt.savefig('./DECLAN_VERIFY1/Kin_v_t.png')
    plt.show()

    plt.plot(time_array, U_lj, '-o', markersize=0.5)
    plt.plot(time_array, lammps_pot, '-o', markersize=0.5)
    plt.ylabel("Potential Energy [kJ/mol]")
    plt.xlabel("Time [fs]")
    plt.savefig('./DECLAN_VERIFY1/LJ_v_t.png')
    plt.show()

    plt.plot(time_array, T, '-o', markersize=0.5)
    plt.plot(time_array, lammps_temperature, '-o', markersize=0.5)
    plt.ylabel("Temperature [K]")
    plt.xlabel("Time [fs]")
    plt.savefig('./DECLAN_VERIFY1/T_v_t.png')
    plt.show()

    plt.plot(time_array, p, '-o', markersize=0.5)
    plt.plot(time_array, lammps_pressure, '-o', markersize=0.5)
    plt.ylabel("Pressure [Pa]")
    plt.xlabel("Time [fs]")
    plt.savefig('./DECLAN_VERIFY1/p_v_t.png')
    plt.show()
    return U_kin, U_lj, T


# 6
def MD_NVT(simulation_time, timestep, T, Q, L, coordinates, force, velocity, zeta, r_cut):
    """
    Function to run an MD-NVT loop for the specified duration and timestep.

    :param simulation_time: Total simulation time [fs]
    :param timestep: Time for each step [fs]
    :param T: Desired system temperature [K]
    :param Q: Damping parameter for thermostat [-]
    :param L: Length of box domain sides [Angstroms]
    :param coordinates: Array containing the coordinates of molecules at a certain time [Angstrom]
    :param force: Array containing the force field of molecules at a certain time [J/Angstrom]
    :param velocity: Array containing the velocities of molecules at a certain time [Angstrom/fs]
    :param zeta: Array containing the 'friction' variable of molecules at a certain time
    :param r_cut: Cut-off distance for inter-molecular interactions [Angstroms]
    :return: State variables kinetic energy, potential energy, temperature and pressure
    """
    start = time.time()

    trajectory_filename = 'trajectory_2.lammps'

    steps = int(simulation_time / timestep)
    sample_frequency = 100
    sample_interval = steps / sample_frequency

    r_old = coordinates
    force_old = force
    v_old = velocity
    zeta_old = zeta
    T_old = T

    U_kin = np.zeros(sample_frequency + 1)
    U_lj = np.zeros(sample_frequency + 1)
    T_array = np.zeros(sample_frequency + 1)
    p = np.zeros(sample_frequency + 1)
    time_array = np.zeros(sample_frequency + 1)

    U_kin[0] = kineticEnergy(v_old)
    U_lj[0] = potentialEnergy(r_old, L, r_cut)
    T_array[0] = T
    p[0] = pressure(T, r_old, L, r_cut)

    counter = 1

    for i in range(1, simulation_time + 1):
        r_new, v_new, force_new, zeta_new, T_new = velocityVerletThermostat(timestep, T_old, Q, r_old, L, force_old,
                                                                            v_old, zeta_old, r_cut)
        print(i)

        if i % sample_interval == 0:
            U_kin[counter] = kineticEnergy(v_new)
            U_lj[counter] = potentialEnergy(r_new, L, r_cut)
            T_array[counter] = T_new
            p[counter] = pressure(T, r_new, L, r_cut)
            time_array[counter] = i
            counter += 1

            write_frame(r_new, L, v_new, force_new, trajectory_filename, (i + 1) * timestep)

        r_old = r_new
        v_old = v_new
        force_old = force_new
        zeta_old = zeta_new
        T_old = T_new

    end = time.time()
    print("The time of execution of above program is :", (end - start) / 60, "mins")
    lammps_temperature, lammps_pressure, lammps_kin, lammps_pot = postProcStateVar(
        './DECLAN_VERIFY2/GIVEN_DATA/log.lammps')

    plt.plot(time_array, U_kin, '-o', markersize=0.5)
    plt.plot(time_array, lammps_kin, '-o', markersize=0.5)
    plt.ylabel("Kinetic Energy [kJ/mol]")
    plt.xlabel("Time [fs]")
    plt.savefig('./DECLAN_VERIFY2/Kin_v_t.png')
    plt.show()

    plt.plot(time_array, U_lj, '-o', markersize=0.5)
    plt.plot(time_array, lammps_pot, '-o', markersize=0.5)
    plt.ylabel("Potential Energy [kJ/mol]")
    plt.xlabel("Time [fs]")
    plt.savefig('./DECLAN_VERIFY2/LJ_v_t.png')
    plt.show()

    plt.plot(time_array, T_array, '-o', markersize=0.5)
    plt.plot(time_array, lammps_temperature, '-o', markersize=0.5)
    plt.ylabel("Temperature [K]")
    plt.xlabel("Time [fs]")
    plt.savefig('./DECLAN_VERIFY2/T_v_t.png')
    plt.show()

    plt.plot(time_array, p, '-o', markersize=0.5)
    plt.plot(time_array, lammps_pressure, '-o', markersize=0.5)
    plt.ylabel("Pressure [Pa]")
    plt.xlabel("Time [fs]")
    plt.savefig('./DECLAN_VERIFY2/p_v_t.png')
    plt.show()

    return U_kin, U_lj, T


"""
# 1.1 Uncomment this to generate rdf plots on a single figure
coordinates_1, l_domain_1 = initGrid(30, 0.5 * 358.4)
coordinates_2, l_domain_2 = initGrid(30,358.4)
coordinates_3, l_domain_3 = initGrid(30, 2 * 358.4)

r, g_r1 = rdf(coordinates_1, np.array([l_domain_1,l_domain_1,l_domain_1]))
g_r2 = rdf(coordinates_2, np.array([l_domain_2,l_domain_2,l_domain_2]))[1]
g_r3 = rdf(coordinates_3, np.array([l_domain_3,l_domain_3,l_domain_3]))[1]

plt.plot(r, g_r1, '-o', markersize = 0.5)
plt.plot(r, g_r2, '-o', markersize = 0.5)
plt.plot(r, g_r3, '-o', markersize = 0.5)
# plt.title("")
plt.ylabel("g(r)")
plt.xlabel("r [A]")
plt.savefig('rdf.png')
plt.show()
"""

"""# 4
# Initializing domain
coord_array, no_of_molecules = initGrid(30, 358.4)
# Initializing velocity
v_array = initVel(150, no_of_molecules)
# Initializing force
force_array = LJ_forces(coord_array, 30, 14)

MD_NVE(3000, 1, 30, coord_array, force_array, v_array, 14)
xyz, vel, F = read_lammps_trj('trajectory_1.lammps')
# xyz, vel, forces = read_lammps_trj('trj.lammps')
# print(xyz)
# print(vel)
# print(F)"""

# 5
# Initializing domain
coord_array, no_of_molecules = initGrid(30, 358.4)
# Initializing velocity
v_array = initVel(150, no_of_molecules)
# Initializing force
force_array = LJ_forces(coord_array, 30, 14)
# Initializing 'friction' variable zeta
zeta_array = np.zeros((no_of_molecules, 3))

MD_NVT(3000, 1, 150, 1e-8, 30, coord_array, force_array, v_array, zeta_array, 14)
xyz, vel, F = read_lammps_trj('trajectory_2.lammps')
# xyz, vel, forces = read_lammps_trj('trj.lammps')
# print(xyz)
# print(vel)
# print(F)
