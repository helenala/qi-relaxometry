import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace, meshgrid, array, abs, ones_like, zeros_like, \
    divide, cos, sin, real, amin, exp, argwhere, diff, sign, all, pi,\
    geomspace, arccos, sqrt

GAMMA = 2 * pi * 28.025e9   # [rad Hz/T]
MU_0 = pi * 4e-7            # vacuum permeability [H/m]
K_B = 1.38064852e-23        # Boltzmann constant [J/K]
ZFS = 2.87e9 * 2 * pi       # [rad Hz] in case of nv with spin=1


class RelaxationRate:
    """This class calculates the relaxation rate. A good approximation of the 
    rate integral requires a pixel resolution of at least:

        x_pixels = 100

        y_pixels = 4000

    This class runs the following functions by default:

        self.create_nv_angles()

        self.create_eigvals_and_eigvecs()

        self.create_esr_frequencies()

        self.create_theta0()

    To calculate the relaxation rate (MHz), run the following functions:

        self.create_k_bounds()

        self.create_k_meshgrids(x_pixels, y_pixels)

        self.create_integrand_grid_exclude_nv_distance()

        self.create_integrand_grid_include_nv_distance()

    The last function creates self.rate_in_MHz.

    All parameters are in SI units.

    Attributes
    ----------
    M_saturation : float
        material constant; magnetisation saturation (default YIG: 142e3 [A/m])
    Gilbert_damping : float
        material constant; Gilbert damping (default YIG: 1e-4 [unitless])
    A_exchange : float
        material exchange constant for spin stiffness (default YIG: 3.7e-12 [J/m])
    temperature : float, int
        temperature (default: 299 [K])
    dir_nv : list or float
        Miller indices in lab frame (default [1,1,1])
    phi_nv : float
        azimuthal angle of the nv (default 0. rad)
    omega : float
        angular frequency (default 0. rad Hz)
    plusmin : int
        +1 or -1, indicating ESR+ or ESR- transition
    bext : float
        external field (default 25e-3 [Tesla])
    theta_bext : float
        external field angle w.r.t. z axis, i.e. film normal (default 54.7 deg)
    distance_nv : float
        distance between nv in diamond tip and the film (default 109e-9 [m])
    film_thickness : float
        film thickness (default 20e-9 [meter])
    quadrants : str
        quadrants in kx-ky plane (default "I+IV")
    zoom_in_heatmap : float
        zoom factor for the heatmap of rate integral's integrand in kx-ky plane 
        (default 1.15)
    """

    def __init__(self,
                 M_saturation=142e3,
                 Gilbert_damping=1e-4,
                 A_exchange=3.7e-12,
                 temperature=299,
                 dir_nv=[1, 1, 1],
                 phi_nv=0.,
                 omega=None,
                 plusmin=-1,
                 bext=25e-3,
                 theta_bext=(54.7/180)*pi,
                 distance_nv=109e-9,
                 film_thickness=20e-9,
                 quadrants="all",
                 zoom_in_heatmap=1.15,
                 ) -> None:
        self.init_locals = locals()  # JSON object of class input arguments
        self.quadrant_factor = 1
        self.plusmin = plusmin
        self.quadrants = quadrants
        self.magnetization_saturation = M_saturation
        self.temperature = temperature  # [Kelvin]
        self.alpha = Gilbert_damping
        self.w_exchange = GAMMA * 2 * A_exchange / \
            self.magnetization_saturation  # [rad Hz m^2]
        self.w_demagnetize = GAMMA * MU_0 * \
            self.magnetization_saturation  # [rad Hz]
        self.dir_nv = dir_nv
        self.omega = omega
        self.bext = bext
        self.theta_bext = theta_bext
        self.distance_nv = distance_nv
        self.film_thickness = film_thickness
        self.zoom_in_heatmap = zoom_in_heatmap
        self.create_nv_angles()
        self.create_eigvals_and_eigvecs()
        self.create_esr_frequencies()
        self.create_theta0()

    def print_input_parameters(self) -> None:
        # Print input arguments in this class
        print(self.init_locals)

    def create_eigvals_and_eigvecs(self) -> None:
        # diagonalize the Hamiltonian
        self.eigval, self.eigvec = np.linalg.eigh(self.create_hamiltonian_nv())

        # sort eigvals from high to low (ESR+,ESR-,0); sort corresp. eigvecs
        self.eigvals = np.sort(self.eigval)[::-1]
        self.eigvecs = self.eigvec[np.argsort(self.eigval)][::-1]

        # eigenvalues
        self.eigval_pls1 = self.eigvals[0]  # |+1>
        self.eigval_min1 = self.eigvals[1]  # |-1>
        self.eigval_0 = self.eigvals[2]  # |0>

        # eigenvectors
        self.eigvec_pls1 = self.eigvecs[0]  # |+1>
        self.eigvec_min1 = self.eigvecs[1]  # |-1>
        self.eigvec_0 = self.eigvecs[2]     # |0>

    def create_bfield_vector(self) -> None:
        """External magnetic field component in nv direction."""
        self.calculate_angle_between_bext_and_nv()
        self.bfield_vector = self.bext * array([sin(self.angle_between_bext_and_nv),
                                                0,
                                                cos(self.angle_between_bext_and_nv)])

    def create_spin_vector(self) -> None:
        """Create 3-vector of Pauli spin 3x3 matrices."""
        # 3x3 Pauli spin matrices.
        spin_x = array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=np.cdouble) / (2 ** 0.5)
        spin_y = array([[0, -1j, 0],
                        [1j, 0, -1j],
                        [0, 1j, 0]], dtype=np.cdouble) / (2 ** 0.5)
        self.spin_z = array([[1, 0, 0],
                             [0, 0, 0],
                             [0, 0, -1]], dtype=np.cdouble)
        # In nv frame.
        self.spin_vector = array(
            [spin_x, spin_y, self.spin_z], dtype=np.cdouble)

    def create_hamiltonian_nv(self) -> np.ndarray:
        """Determine the Hamiltonian to calculate the ESR frequencies:
        ZFS * Sz^2 + GAMMA * dot(B, S), where ZFS stands for zero-field splitting.

        Attributes
        ----------
        B_dot_S : np.ndarray
            inner product between magnetic field and spin vector

        Returns
        -------
        ZFS * S_Z**2 + GAMMA * B_dot_S : float
            Hamiltonian for the spin-1 nv.
        """
        self.create_bfield_vector()
        self.create_spin_vector()
        B_dot_S = np.tensordot(self.bfield_vector, self.spin_vector, axes=1)
        return ZFS * self.spin_z**2 + GAMMA * B_dot_S

    def calculate_angle_between_bext_and_nv(self) -> None:
        """Calculates angle between external magnetic field (bext) vector and 
        nitrogen-vacancy (nv) axis. If bext and nv are parallel, then the angle
        (angle_between_bext_and_nv) between bext and nv is zero. If bext and nv are not
        parallel, then the angle is arccos(bext * nv), where bext en nv are
        normalized.
        """
        dir_bext = array([sin(self.theta_bext),
                          0,
                          cos(self.theta_bext)], dtype=float)  # length 1
        if (np.array_equal(dir_bext, self.dir_nv)):
            # bext parallel to nv
            self.angle_between_bext_and_nv = 0.
        else:
            # bext not parallel to nv
            self.angle_between_bext_and_nv = np.arccos(
                np.dot(self.dir_nv, dir_bext))

    def create_esr_frequencies(self) -> None:
        """ESR freq only appears in susceptibility_lambda() and 
        susceptibility_matrix().
        """

        # freq differences
        self.esr_pls = self.eigval_pls1 - self.eigval_0
        self.esr_min = self.eigval_min1 - self.eigval_0

        if (self.omega != None):
            self.w = self.omega
        else:
            if (self.plusmin == 1):
                self.w = self.esr_pls
            elif (self.plusmin == -1):
                self.w = self.esr_min
            else:
                print("No ESR frequency calculated or found!")

    def create_nv_angles(self) -> None:
        """In this model, we rotate the standard NV orientations by phi = -pi/4 
        around the z axis such that the NV orientations [111] and [-1-11] are 
        placed in the x-z plane, whereas [1-1-1] and [-11-1] are placed in y-z 
        plane. 
        """
        if (type(self.dir_nv) == float):
            self.theta_nv, self.phi_nv = self.dir_nv, self.init_locals["phi_nv"]
        elif (self.dir_nv == [1, 1, 1]):
            self.theta_nv, self.phi_nv = arccos(1/sqrt(3)), 0.
        elif (self.dir_nv == [-1, -1, 1]):
            self.theta_nv, self.phi_nv = arccos(1/sqrt(3)), pi
        elif (self.dir_nv == [1, -1, -1]):
            self.theta_nv, self.phi_nv = arccos(-1/sqrt(3)), pi/2
        elif (self.dir_nv == [-1, 1, -1]):
            self.theta_nv, self.phi_nv = arccos(-1/sqrt(3)), -pi/2
        else:
            print("dir_nv must be w.r.t. either an angle in radians or a list" +
                  "in the form: \n\t[1,1,1], [-1,-1,1], [1,-1,-1] or [-1,1,-1].")
        self.dir_nv = array([sin(self.theta_nv) * cos(self.phi_nv),
                             sin(self.theta_nv) * sin(self.phi_nv),
                             cos(self.theta_nv)])  # length 1
        self.cos_phi_nv = cos(self.phi_nv)
        self.sin_phi_nv = sin(self.phi_nv)
        self.cos_theta_nv = cos(self.theta_nv)
        self.sin_theta_nv = sin(self.theta_nv)

    def create_theta0(self) -> None:
        """Calculate the magnet film's magnetization angle t0 using eq:

        bext * sin(t0 - tb) = .5 * MU_0 * M_s * sin(2 * t0),

        where tb is the angle of the external magnetic field bext, MU_0 is the
        magnetic permeability constant in vacuum, and M_s is the magnetization
        saturation.

        Returns
        -------
        t0 : float
            angle [rad] of the magnet film's magnetization
        """
        t0 = linspace(0, pi, 20000)
        tb = self.theta_bext * ones_like(t0)
        rhs = .5 * MU_0 * self.magnetization_saturation * sin(2 * t0)
        lhs = self.bext * sin(t0 - tb)
        # index of theta0 at intersection between lhs and rhs
        i = argwhere(diff(sign(lhs - rhs))).flatten()[0]
        if all(lhs == 0.):
            self.theta0 = pi/2
        else:
            self.theta0 = t0[i]
        self.cos_theta0 = cos(self.theta0)
        self.sin_theta0 = sin(self.theta0)

    def fmr(self):
        """Ferromagnetic resonance frequency (Hz)."""
        B = self.bext * sin(self.theta_bext)  # in-plane component of bext
        return GAMMA * sqrt(B * (B + MU_0 * self.magnetization_saturation))

    def create_k_bounds(self,
                        kx_min=-9e6*2*pi, kx_max=9e6*2*pi,
                        ky_min=-9e6*2*pi, ky_max=9e6*2*pi
                        ) -> None:
        """Coarse grid of 500x500 pixels to determine ky bound."""
        self.create_quadrants()
        self.kx_min = kx_min
        self.kx_max = kx_max
        self.ky_min = ky_min
        self.ky_max = ky_max
        self.create_k_meshgrids()
        self.create_w_meshgrids()
        fig, axes = plt.subplots(nrows=2)
        cs = axes[0].contour(self.kx,
                             self.ky,
                             self.omega_spin_wave_dispersion(),
                             levels=geomspace(self.w, self.w * 2, 10)
                             )
        plt.close()
        isofrequencies = []
        if len(cs.collections) > 1:
            for i in range(len(cs.collections)):
                lines = []
                for line in cs.collections[i].get_paths():
                    lines.append(line.vertices)  # numpy array
                    x = lines[0][:, 0]
                    y = lines[0][:, 1]
                    isofrequencies.append([x, y])
            kx_iso = isofrequencies[0][0]
            ky_iso = isofrequencies[0][1]
            self.kx_max = abs(amin(kx_iso)) * self.kx2 * self.zoom_in_heatmap
            self.ky_min = amin(ky_iso) * self.ky1 * self.zoom_in_heatmap
            self.kx_min = -self.kx_max * self.kx1
            self.ky_max = -self.ky_min * self.ky2
        else:
            print("cs.collections =< 1")
            self.ky_min = -2e7 * self.ky1 * self.zoom_in_heatmap
            self.kx_max = 2e7 * self.kx2 * self.zoom_in_heatmap
            self.kx_min = -self.kx_max * self.kx1
            self.ky_max = -self.ky_min * self.ky2

    def create_quadrants(self) -> None:
        """Choosing specific quadrants may reduce the number of calculations of
        pixels due to symmetry (in kx).
        """
        if (self.quadrants == "I"):
            self.create_qfactor(0, 1, 0, 1)
        elif (self.quadrants == "II"):
            self.create_qfactor(1, 0, 0, 1)
        elif (self.quadrants == "III"):
            self.create_qfactor(1, 0, 1, 0)
        elif (self.quadrants == "IV"):
            self.create_qfactor(0, 1, 1, 0)
        elif (self.quadrants == "I+IV"):
            self.quadrant_factor = 2
            self.create_qfactor(0, 1, 1, 1)
        elif (self.quadrants == "II+III"):
            self.quadrant_factor = 2
            self.create_qfactor(1, 1, 1, 1)
        else:
            self.quadrant_factor = 1
            self.create_qfactor(1, 1, 1, 1)

    def create_qfactor(self, kx1, kx2, ky1, ky2) -> None:
        self.kx1, self.kx2, self.ky1, self.ky2 = kx1, kx2, ky1, ky2

    def create_k_meshgrids(self, x_pixels=800, y_pixels=800) -> None:
        """Create kx and ky meshgrids based on the calculated k bounds. The
        following objects are generated:

            self.kx : numpy meshgrid 
                2D array of kx points (default 800x800)
            self.ky : numpy meshgrid
                2D array of ky points (default 800x800)

        Attributes
        ----------
        x_pixels : int
            number of pixels in kx direction in kx-ky plane (default 800)
        y_pixels : int
            number of pixels in ky direction in kx-ky plane (default 800)
        """
        kx = linspace(self.kx_min, self.kx_max, x_pixels)
        ky = linspace(self.ky_min, self.ky_max, y_pixels)
        self.kx, self.ky = meshgrid(kx, ky)

    def create_w_meshgrids(self) -> None:
        """Create kx and ky meshgrids based on the calculated k bounds. The
        following objects are generated:

            self.k : numpy meshgrid 
                2D array of k points (default 800x800)
            self.fk : numpy meshgrid 
                2D array of fk =  points (default 800x800)
            self.cos_phi_k : numpy meshgrid
                2D array of kx/k points (default 800x800)
            self.sin_phi_k : numpy meshgrid
                2D array of ky/k points (default 800x800)
            self.w0 : numpy meshgrid
                2D array of omega_0 points (default 800x800)
            self.w1 : numpy meshgrid
                2D array of omega_1 points (default 800x800)
            self.w2 : numpy meshgrid
                2D array of omega_2 points (default 800x800)
            self.w3 : numpy meshgrid
                2D array of omega_3 points (default 800x800)

        """
        self.k = ((self.kx)**2 + (self.ky)**2) ** 0.5
        self.cos_phi_k = divide(self.kx,
                                self.k,
                                out=zeros_like(self.kx),
                                where=(self.k != 0))
        self.sin_phi_k = divide(self.ky,
                                self.k,
                                out=zeros_like(self.ky),
                                where=(self.k != 0))
        self.fk = self.f_k()
        self.w0 = self.omega_0()
        self.w1 = self.omega_1()
        self.w2 = self.omega_2()
        self.w3 = self.omega_3()

    def create_correlations_components(self):
        """For k -> -k and w -> -w, only w and w1 obtain a minus sign."""
        self.susceptibility_pls_k_pls_w = self.susceptibility_matrix(self.w,
                                                                     self.w1)
        self.susceptibility_min_k_min_w = self.susceptibility_matrix(-self.w,
                                                                     -self.w1)

    def susceptibility_matrix(self, w, w1) -> np.ndarray:
        """Spin matrix [rad Hz] without prefactor gamma/Lambda."""
        self.i_alpha_w = 1j * self.alpha * w
        S_11 = self.w3 - self.i_alpha_w
        S_12 = -w1 - 1j * w
        S_21 = -w1 + 1j * w
        S_22 = self.w2 - self.i_alpha_w
        self.susceptibility_matrix_exclude_gamma_over_lambda = array([[S_11, S_12],
                                                                      [S_21, S_22]],
                                                                     dtype=np.cdouble)
        gamma_over_lambda = GAMMA * self.one_over_susceptibility_lambda()
        return gamma_over_lambda * self.susceptibility_matrix_exclude_gamma_over_lambda

    def one_over_susceptibility_lambda(self):
        susceptibility_lambda = self.susceptibility_lambda()
        return divide(ones_like(susceptibility_lambda),
                      susceptibility_lambda,
                      out=zeros_like(susceptibility_lambda),
                      where=(susceptibility_lambda != 0))

    def susceptibility_lambda(self):
        """Generate the following function:

            (w2 - i * alpha * w) * (w3 - i * alpha * w) - (w1)^2 - w^2,

        where w is shorthand notation for angular frequency 'omega'.

        Returns
        -------
        Lambda : numpy array or float
            prefactor of the susceptibility matrix (default 800x800 
            meshgrid)
        """
        term1 = (self.w2 - self.i_alpha_w) * (self.w3 - self.i_alpha_w)
        term2 = - (self.w1)**2 - (self.w)**2
        return term1 + term2

    def f_k(self):
        """Generate the following exponential function:

            1 - (1 - exp(-k*L)) / (k*L)

        Returns
        -------
        f_k : numpy array or float
            exponential function in the susceptibility matrix (default 800x800 
            meshgrid)
        """
        den = self.k * self.film_thickness  # k*L
        nom = den - 1 + exp(-den)
        # matrix element must be 0 if divide by den=0
        return divide(nom, den, out=zeros_like(nom), where=(den != 0))

    def omega_0(self) -> np.ndarray:
        """Generate the following exponential function:

            omega_b * cos(t0 - tB) - omega_d * cos(t0)^2 + omega_ex * k^2,

        where:

            t0 : float
                magnetic film's angle of magnetization (theta0)
            tB : float
                angle of the external magnetic field (bext)
            omega_b = gamma * bext : float
                Zeeman term
            omega_d = gamma * mu_0 * M_s : float
                demagnetizing term
            omega_ex = 2 * A_exchange / M_s : float
                exchange (spin stiffness) term

        All parameters are in SI units.

        Returns
        -------
        f_k : numpy array or float
            exponential function in the susceptibility matrix (default 800x800 
            meshgrid)
        """
        w_internal = GAMMA * self.bext * cos(self.theta0 - self.theta_bext)
        w_internal -= self.w_demagnetize * self.cos_theta0**2
        return w_internal + self.w_exchange * (self.k)**2

    def omega_1(self) -> np.ndarray:
        return self.w_demagnetize * self.fk * self.sin_phi_k * \
            self.cos_phi_k * self.cos_theta0

    def omega_2(self) -> np.ndarray:
        term1 = self.fk * self.cos_phi_k**2 * self.cos_theta0**2
        term2 = (1 - self.fk) * (self.sin_theta0**2)
        return self.w0 + self.w_demagnetize * (term1 + term2)

    def omega_3(self) -> np.ndarray:
        return self.w0 + self.w_demagnetize * self.fk * self.sin_phi_k**2

    def omega_spin_wave_dispersion(self) -> np.ndarray:
        return ((self.w2 * self.w3) - (self.w1 ** 2)) ** 0.5

    def calculate_relaxation_rate_in_MHz(self, x_pixels=800, y_pixels=800):
        self.create_k_bounds()
        self.create_k_meshgrids(x_pixels, y_pixels)
        self.calculate_sum_di_dj_cij()
        self.create_integrand_grid_exclude_nv_distance()
        self.create_integrand_grid_include_nv_distance()

    def create_integrand_grid_include_nv_distance(self) -> np.ndarray:
        """Numpy 2D array relaxation rate integrand (MHz)."""
        dkx = abs(self.kx[0][0] - self.kx[0][1])
        dky = abs(self.ky[0][0] - self.ky[1][0])
        rate_grid = self.integrand_grid_exclude_nv_distance * dkx * dky
        rate_grid *= (-1/2 * exp((-self.k) * self.distance_nv)) ** 2  # [MHz]
        self.integrand_grid_include_nv_distance = rate_grid
        self.rate_in_MHz = np.sum(rate_grid)

    def create_integrand_grid_exclude_nv_distance(self) -> None:
        """Calculate the integrand meshgrid without distance factor exp(-2d*k).
        Run the following commands prior to this function:

            create_k_bounds()

            create_k_meshgrids(x_pixels, y_pixels)

        These two functions are needed to create meshgrids for kx and ky with
        the correct bounds.

        Parameters
        ----------
        pixel_area : float
            area [(2pi/m)^2] of one square pixel in the heatmap, we need this 
            in the Riemann sum approximation of the rate integral since dk is 
            approximately delta_kx * delta_ky
        """
        integrand = self.sum_di_dj_cij

        # multiply by 2*D_thermal [T^2 m^2 s] from c_ij [m^2 s]
        integrand *= 2 * self.alpha * K_B * self.temperature / \
            (GAMMA * self.magnetization_saturation * self.film_thickness)

        integrand *= (GAMMA**2 / 2) * (2 * pi)**(-2) * 1e-6  # Hz to MHz
        integrand *= self.quadrant_factor
        integrand *= (MU_0 * self.magnetization_saturation *
                      (1 - exp(-self.k * self.film_thickness)))**2
        self.integrand_grid_exclude_nv_distance = real(integrand)  # [MHz]

    def calculate_sum_di_dj_cij(self) -> None:
        self.create_w_meshgrids()
        self.create_correlations_components()
        self.dipolar_tensor()
        self.di_dj_cij()
        self.sum_di_dj_cij = self.Dx_Dx_Cxx + \
            self.Dx_Dy_Cxy + self.Dy_Dx_Cyx + self.Dy_Dy_Cyy

    def dipolar_tensor(self) -> None:
        self.cos_phi_k = divide(
            self.kx, self.k, out=zeros_like(self.kx), where=(self.k != 0))
        self.sin_phi_k = divide(
            self.ky, self.k, out=zeros_like(self.ky), where=(self.k != 0))

        self.d_xx_pls_k = self.d_xx()
        self.d_xy_pls_k = self.d_xy()
        self.d_yx_pls_k = self.d_yx()
        self.d_yy_pls_k = self.d_yy()

        self.cos_phi_k *= (-1)
        self.sin_phi_k *= (-1)
        self.d_xx_min_k = self.d_xx()
        self.d_xy_min_k = self.d_xy()
        self.d_yx_min_k = self.d_yx()
        self.d_yy_min_k = self.d_yy()

        self.Dx_pls_k = self.d_xx_pls_k - self.plusmin * 1j * self.d_yx_pls_k
        self.Dy_pls_k = self.d_xy_pls_k - self.plusmin * 1j * self.d_yy_pls_k
        self.Dx_min_k = self.d_xx_min_k + self.plusmin * 1j * self.d_yx_min_k
        self.Dy_min_k = self.d_xy_min_k + self.plusmin * 1j * self.d_yy_min_k

    def di_dj_cij(self) -> None:
        self.Dx_Dx_Cxx = self.Dx_pls_k * self.Dx_min_k * self.c_ij(0, 0)
        self.Dx_Dy_Cxy = self.Dx_pls_k * self.Dy_min_k * self.c_ij(0, 1)
        self.Dy_Dx_Cyx = self.Dy_pls_k * self.Dx_min_k * self.c_ij(1, 0)
        self.Dy_Dy_Cyy = self.Dy_pls_k * self.Dy_min_k * self.c_ij(1, 1)

    def d_xx(self) -> np.ndarray:
        term1 = self.cos_phi_k ** 2 * self.cos_theta0 - \
            1j * self.cos_phi_k * self.sin_theta0
        term2 = self.cos_phi_k * self.sin_phi_k * \
            self.cos_theta0 - 1j * self.sin_phi_k * self.sin_theta0
        term12 = self.cos_phi_nv * term1 - self.sin_phi_nv * term2
        term3 = 1j * self.cos_phi_k * self.cos_theta0 + self.sin_theta0
        return self.cos_theta_nv * term12 - self.sin_theta_nv * term3

    def d_xy(self) -> np.ndarray:
        term1 = self.cos_phi_nv * self.cos_phi_k * self.sin_phi_k
        term2 = self.sin_phi_nv * self.sin_phi_k ** 2
        return self.cos_theta_nv * (term1 - term2) - \
            self.sin_theta_nv * 1j * self.sin_phi_k

    def d_yx(self) -> np.ndarray:
        term1 = self.cos_phi_k ** 2 * self.cos_theta0 - \
            1j * self.cos_phi_k * self.sin_theta0
        term2 = self.cos_phi_k * self.sin_phi_k * \
            self.cos_theta0 - 1j * self.sin_phi_k * self.sin_theta0
        return self.sin_phi_nv * term1 + self.cos_phi_nv * term2

    def d_yy(self) -> np.ndarray:
        return self.sin_phi_nv * self.cos_phi_k * self.sin_phi_k + \
            self.cos_phi_nv * self.sin_phi_k ** 2

    def c_ij(self, i, j) -> np.ndarray:
        """Magnetization correlations c_ij / (2*D_thermal) [T^-2 ]. Units:
            * c_ij [m^2 s].
            * D_thermal [T^2 m^2 s].
        Factor 2*D_thermal will be included in 
        create_integrand_grid_exclude_nv_distance().
        """
        SixSjx = self.susceptibility_pls_k_pls_w[i][0] * \
            self.susceptibility_min_k_min_w[j][0]  # [1/T]
        SiySjy = self.susceptibility_pls_k_pls_w[i][1] * \
            self.susceptibility_min_k_min_w[j][1]  # [1/T]
        return SixSjx + SiySjy
