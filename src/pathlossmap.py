def compute_pathloss_map():
    E_total = E_LOS + E_diff + E_g_ref
    # Compute power density S (W/m^2);
    S = (abs(E_total ) ** 2) / 377.0
    
    # Effective aperture for an omni antenna (m^2)
    A_e = lambda_ ** 2 / (4 * np.pi)
    
    # Received power (W); avoid log(0) by applying a minimum floor
    P_total = S * A_e
    #P_total = np.maximum(P_total, 1e-30)
    # Convert received power to dBm\n
    Pr_total_dBm = 10 * np.log10(P_total) + 30
    Pr_total_dBm = Pr_total_dBm.reshape(ny,nx)
    PL_total = P_t_dBm - Pr_total_dBm
    PL_total = PL_total.reshape(ny,nx)
    distances = np.linalg.norm(R_grid - T, axis=1)
    NLOS_path_loss = compute_ci_path_loss(distances)
    PL_total = PL_total.flatten()
    PL_total = np.where(np.isinf(PL_total), NLOS_path_loss, PL_total)
    PL_total = PL_total.reshape(ny,nx)
    BEL = calc_BEL()
    PL_total = PL_total.flatten()
    PL_total[~valid_rx_mask] = PL_total[~valid_rx_mask] + BEL[~valid_rx_mask]
    PL_total = PL_total.reshape(ny,nx)
    PL_smoothed = smooth_pathloss(PL_total)
    return PL_smoothed


def compute_ci_path_loss(distances, path_loss_exponent=3.0, sigma=6.8, reference_distance=1.0):
    """
    Compute the Close-In (CI) path loss with shadow fading.
    K. Haneda et al., "5G 3GPP-Like Channel Models for Outdoor Urban Microcellular and Macrocellular Environments,"
    2016 IEEE 83rd Vehicular Technology Conference (VTC Spring), Nanjing, China, 2016, pp. 1-7,
    doi: 10.1109/VTCSpring.2016.7503971.


    Parameters:
    - distances: Array of distances between transmitter and receivers in meters.
    - path_loss_exponent: Path loss exponent (n).
    - sigma: Standard deviation of shadow fading in dB.
    - reference_distance: Reference distance d_0 in meters (default is 1 meter).

    Returns:
    - Array of path loss values in dB for each receiver point.
    """

    # Free-space path loss at the reference distance
    fspl_d0 = 20 * np.log10(4 * np.pi * reference_distance / lambda_)

    # Path loss at each distance without shadow fading
    path_loss = fspl_d0 + 10 * path_loss_exponent * np.log10(distances / reference_distance)

    # Shadow fading component
    shadow_fading = np.random.normal(0, sigma, size=distances.shape)

    # Total path loss with shadow fading
    total_path_loss = path_loss + shadow_fading

    return total_path_loss

# Function to calculate BEL (Building Entry Loss)

def calc_BEL(frequency=28, probability=0.5):
    """
    Calculates the Building Entry Loss (BEL) for a given frequency and probability.

    This function computes the BEL based on statistical models that incorporate
    median horizontal path loss, elevation angle correction, and probabilistic
    variations in loss due to building penetration.

    Parameters:
    -----------
    frequency : float, optional
        The frequency in GHz for which the BEL is to be calculated. Default is 28 GHz.
    probability : float, optional
        The probability level (between 0 and 1) used to compute the inverse
        cumulative distribution function (CDF) for log-normal distributions.
        Default is 0.5 (median value).

    Returns:
    --------
    np.ndarray
        An array of computed Building Entry Loss (BEL) values in dB,
        having the same length as R_horiz.

    """

    horizontal_dist = np.linalg.norm(R_horiz - T_horiz, axis=1)
    Phi = np.arctan2(T[2], horizontal_dist) * (180.0 / np.pi)

    r, s, t = 12.64, 3.72, 0.96  # Coefficients for L_h
    u, v = 9.6, 2.0              # σ1 parameters
    w, x = 9.1, -3.0             # μ2 parameters
    y, z = 9.4, -2.1             # σ2 parameters
    C = -3.0                     # Constant for BEL calculation
    F_inv_p = norm.ppf(probability)  # Calculate F^-1(P)

    # Calculate median horizontal path loss (scalar)
    L_h = r + s * np.log10(frequency) + t * (np.log10(frequency))**2

    # Elevation angle correction (array)
    L_e = 0.212 * np.abs(Phi)

    # Calculate μ1 (array) and μ2 (scalar)
    mu_1 = L_h + L_e
    mu_2 = w + x * np.log10(frequency)  # Scalar

    # Calculate σ1 and σ2 (both scalars)
    sigma_1 = u + v * np.log10(frequency)
    sigma_2 = y + z * np.log10(frequency)

    # Calculate A(P) and B(P) (A_P is an array, B_P is broadcasted to an array)
    A_P = F_inv_p * sigma_1 + mu_1
    B_P = F_inv_p * sigma_2 + mu_2

    # Calculate Building Entry Loss (BEL) as an array
    BEL = 10 * np.log10(10**(0.1 * A_P) + 10**(0.1 * B_P) + 10**(0.1 * C))

    return BEL

def smooth_pathloss(pathloss_2d):
    """
    Applies a 3×3 averaging filter to smooth a 2D pathloss map.

    This function performs a convolution operation where each pixel is replaced
    by the average of itself and its neighboring pixels. Edge pixels are handled
    properly by averaging only available neighbors.

    Parameters:
    -----------
    pathloss_2d : np.ndarray
        A 2D array representing pathloss values over a geographic area.

    Returns:
    --------
    np.ndarray
        A smoothed 2D pathloss map with the same shape as the input.
    """

    # Define a 3×3 averaging kernel
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]) / 5.0

    # Apply 2D convolution
    smoothed_pathloss = convolve(pathloss_2d, kernel, mode='nearest')

    return smoothed_pathloss
