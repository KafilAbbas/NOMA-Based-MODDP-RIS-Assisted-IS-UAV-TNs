import numpy as np
from scipy.special import j1  # Bessel function of first kind
# ==============================
# Common Parameters
# ==============================
carrier_freq = 2e9  # 2 GHz
wavelength = 3e8 / carrier_freq
num_ris_elements = 64  # M elements (M_R x N_R = 8x8)
M_R = 8  # Vertical elements
N_R = 8  # Horizontal elements
d_x = wavelength/2  # Horizontal spacing
d_y = wavelength/2  # Vertical spacing

# ==============================
# 1. Satellite to VU Channel (G_SV)
# ==============================


def satellite_beam_gain_matrix(num_beams, num_users, beam_width=0.8, max_gain=52.1):
    """
    Implements the beam gain matrix based on reference [47]
    
    Parameters:
    num_beams : number of beams (K in paper)
    num_users : number of users (N in paper)
    beam_width : 3dB beamwidth in degrees (default 0.8° for GEO satellites)
    max_gain : maximum beam gain in dBi (default 52.1dBi for typical GEO)
    
    Returns:
    b : beam gain matrix of shape (num_beams, num_users)
    """
    # Convert from dBi to linear scale
    G_max = 10**(max_gain / 10)
    
    # Generate random user angles within beam coverage
    # (In practice, these would be the actual user angles)
    theta_users = np.random.uniform(-beam_width*3, beam_width*3, num_users)
    
    # Initialize beam gain matrix
    b = np.zeros((num_beams, num_users))
    
    # Satellite beam angles (equally spaced for simplicity)
    theta_beams = np.linspace(-beam_width*3, beam_width*3, num_beams)
    
    # Calculate beam pattern for each beam-user pair
    for k in range(num_beams):
        for n in range(num_users):
            # Angle difference between beam center and user
            theta = theta_users[n] - theta_beams[k]
            
            # Normalized angle (Eq.3 in [47])
            u = 2.07123 * np.sin(np.deg2rad(theta)) / np.sin(np.deg2rad(beam_width/2))
            
            # Beam pattern (using J1 Bessel function)
            if u == 0:
                b[k,n] = G_max
            else:
                b[k,n] = G_max * (j1(u)/u)**2
    
    return b

def satellite_to_vu_channel(theta_s, d_s, G_s_t ,G_s_r_Max, antenna_diam, mu_rain, sigma_rain):
    """
    Parameters:
    theta_s : off-axis angle (degrees)
    d_s : distance between satellite and VU (meters)
    G_s_t : satellite launch gain
    G_s_r_Max : maximum satellite receive gain
    antenna_diam : receiver antenna diameter (meters)
    mu_rain, sigma_rain : rain attenuation parameters
    """
    # Calculate G_s_r (equation 2)
    lambda_wave = wavelength
    theta_a = (20*lambda_wave/antenna_diam) * np.sqrt(G_s_r_Max - (2 + 15*np.log10(antenna_diam/lambda_wave)))
    theta_b = 15.85*(antenna_diam/lambda_wave)**-0.6
    
    if 0 < theta_s < theta_a:
        G_s_r = G_s_r_Max - 2.5e-3 * (antenna_diam * theta_s / lambda_wave)**2
    elif theta_a <= theta_s < theta_b:
        G_s_r = 2 + 15*np.log10(antenna_diam/lambda_wave)
    elif theta_b <= theta_s < 48:
        G_s_r = 32 - 25*np.log10(theta_s)
    else:
        G_s_r = -10
    
    # Free space loss
    C_s = (lambda_wave/(4*np.pi*d_s))**2
    
    # Rain attenuation (lognormal in dB)
    xi_dB = np.random.lognormal(mean=mu_rain, sigma=sigma_rain)
    xi = 10**(xi_dB/10)
    
    # Beam gain (simplified circular pattern)
    # (Actual implementation would use specific antenna pattern)
    b = satellite_beam_gain_matrix(num_beams=1, num_users=1)  # Placeholder
    beam_gain = b[0,0]  # Get gain for this beam-user pair
    # Phase component
    phase = np.exp(-1j*2*np.pi*d_s/lambda_wave)
    
    # Combined channel
    G = np.sqrt(G_s_t * G_s_r * C_s)  * (xi**-0.5) * beam_gain**0.5 * phase
    return G

# ==============================
# 2. RIS to VU Channel (h_RV)
# ==============================
def ris_to_vu_channel(K_RE, theta_r, phi_r, G_e, d_ris_vu):
    """
    Parameters:
    K_RE : Rician factor
    theta_r : elevation angle (radians)
    phi_r : azimuth angle (radians)
    G_e : VU receive gain
    d_ris_vu : distance between RIS and VU (meters)
    """
    # Free space loss
    C_e = (wavelength/(4*np.pi*d_ris_vu))**2
    
    # Steering vectors (equations 6-7)
    a_x = np.exp(1j*2*np.pi*d_x/wavelength * np.sin(theta_r)*np.cos(phi_r) * np.arange(M_R))
    a_y = np.exp(1j*2*np.pi*d_y/wavelength * np.sin(theta_r)*np.sin(phi_r) * np.arange(N_R))
    
    A = np.outer(a_x, np.conj(a_y))  # Outer product
    h_LOS = np.sqrt(G_e * C_e) * A.flatten()
    
    # NLOS component
    h_NLOS = (np.random.randn(num_ris_elements) + 
              1j*np.random.randn(num_ris_elements)) / np.sqrt(2)
    
    # Combine components
    h_RE = (np.sqrt(K_RE/(K_RE + 1)) * h_LOS + 
            np.sqrt(1/(K_RE + 1)) * h_NLOS)
    return h_RE

# ==============================
# 3. Satellite to RIS Channel (G_SR)
# ==============================
def satellite_to_ris_channel(d_sr, f_c, G_s_t_ris_dB):
    """
    Satellite to RIS channel G_SR.
    d_sr: distance between satellite and RIS (meters)
    f_c: carrier frequency (Hz)
    G_s_t_ris_dB: satellite transmit gain towards RIS (in dB)
    """
    c = 3e8  # Speed of light
    lambda_wave = c / f_c

    # Free space path loss
    C_sr = (lambda_wave / (4 * np.pi * d_sr))**2
    
    # Random phase
    phase = np.exp(-1j * 2 * np.pi * d_sr / lambda_wave)
    
    # Rain attenuation (Normal distribution in dB)
    xi_dB = np.random.normal(loc=0.1, scale=0.3)  # Example parameters
    xi = 10 ** (xi_dB / 10)
    
    # Transmit gain (convert from dB to linear)
    G_s_t_ris = 10 ** (G_s_t_ris_dB / 10)
    
    # Final channel
    G_SR = np.sqrt(G_s_t_ris) * C_sr * xi**-0.5 * phase
    
    return G_SR


# ==============================
# Example Usage
# ==============================
if __name__ == "__main__":
    # Satellite to VU example
    G_SV = satellite_to_vu_channel(
        theta_s=5,  # degrees
        d_s=500e3,  # 500 km
        G_s_t=30,   # dBi
        antenna_diam=0.5,  # meters
        mu_rain=0.1,
        sigma_rain=0.3
    )
    
    # RIS to VU example
    h_RV = ris_to_vu_channel(
        K_RE=10,
        theta_r=np.deg2rad(30),
        phi_r=np.deg2rad(45),
        G_e=3,       # dBi
        d_ris_vu=1e3 # 1 km
    )
    
    # Satellite to RIS example
    G_SR = satellite_to_ris_channel(
        d_sr=501e3,     # 501 km
        G_s_t_ris=30,   # dBi
        antenna_diam_ris=0.5
    )
    
    print("Satellite-to-VU Channel:", G_SV)
    print("RIS-to-VU Channel Shape:", h_RV.shape)
    print("Satellite-to-RIS Channel:", G_SR)