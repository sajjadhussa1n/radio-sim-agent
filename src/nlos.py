import numpy as np

def compute_ci_path_loss(T, R_grid, path_loss_exponent=3.0, sigma=6.8, reference_distance=1.0):
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

    distances = np.linalg.norm(R_grid - T, axis=1)

    # Free-space path loss at the reference distance
    fspl_d0 = 20 * np.log10(4 * np.pi * reference_distance / lambda_)

    # Path loss at each distance without shadow fading
    path_loss = fspl_d0 + 10 * path_loss_exponent * np.log10(distances / reference_distance)

    # Shadow fading component
    shadow_fading = np.random.normal(0, sigma, size=distances.shape)

    # Total path loss with shadow fading
    total_path_loss = path_loss + shadow_fading

    return total_path_loss
