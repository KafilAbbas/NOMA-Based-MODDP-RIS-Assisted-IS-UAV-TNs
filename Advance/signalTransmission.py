import numpy as np

# Parameters
K = 4       # Number of users
M = 32      # RIS elements
P_max = 10  # Max transmit power (W)
sigma2 = 1e-9  # Noise variance

# Example channel matrices (randomly initialized for demonstration)
G_Sk = np.random.randn(K, M) + 1j * np.random.randn(K, M)  # Satellite to users
h_Rk = np.random.randn(K, M) + 1j * np.random.randn(K, M)  # RIS to users
G_SR = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)  # Satellite to RIS

# RIS phase shift matrix (diagonal)
theta = np.random.uniform(0, 2*np.pi, M)
Theta = np.diag(np.exp(1j * theta))  # RIS configuration

# Beamforming vectors (ZF precoding for simplicity)
# Assume satellite uses ZF to nullify interference
H_eff = np.array([G_Sk[k] + h_Rk[k] @ Theta @ G_SR for k in range(K)])
W = np.linalg.pinv(H_eff)  # ZF precoding matrix

# Power allocation (equal power for OMA)
p = np.ones(K) * (P_max / K)  # Power per user

def calculate_OMA_rate():
    rates = []
    for k in range(K):
        # Effective channel for user k
        h_k = G_Sk[k] + h_Rk[k] @ Theta @ G_SR
        
        # Signal power
        signal = np.abs(h_k @ W[:, k])**2 * p[k]
        
        # Interference from other users
        interference = sum(
            np.abs(h_k @ W[:, j])**2 * p[j] for j in range(K) if j != k
        )
        
        # SINR calculation
        sinr = signal / (interference + sigma2)
        rates.append(np.log2(1 + sinr))
    
    return sum(rates)  # Sum rate

print(f"OMA System Sum Rate: {calculate_OMA_rate():.2f} bps/Hz")


def calculate_NOMA_rate():
    # Sort users by channel gain (descending: strongest to weakest)
    h_eff = [G_Sk[k] + h_Rk[k] @ Theta @ G_SR for k in range(K)]
    gains = [np.linalg.norm(h) for h in h_eff]
    sorted_users = np.argsort(gains)[::-1]  # Descending order
    
    # Power allocation: weaker users get more power
    alpha = 1  # Increased from 0.6 to reduce power disparity
    power_coefficients = [alpha ** (K - 1 - i) for i in range(K)]
    total_power = sum(power_coefficients)
    p = np.array([(P_max * coeff) / total_power for coeff in power_coefficients])
    
    # Validate SIC constraints (equation 13)
    rho_min = 1e-3  # Reduced threshold for demonstration
    for i in range(1, K):
        # Previous user (stronger) and current user (weaker)
        h_prev = h_eff[sorted_users[i-1]]
        h_prev_norm = np.linalg.norm(h_prev)**2
        
        # Ensure weaker user's power > sum of stronger users' powers
        term = p[i] * h_prev_norm - sum(p[:i]) * h_prev_norm
        if term < rho_min:
            # Dynamically adjust power to meet constraint
            p[i] = (sum(p[:i]) * h_prev_norm + rho_min) / h_prev_norm
    
    # Re-normalize power to ensure total <= P_max
    p = p * (P_max / np.sum(p))
    
    # Calculate rates
    rates = []
    B = 1e6  # Bandwidth (Hz)
    for i, user in enumerate(sorted_users):
        h_i = h_eff[user]
        h_i_norm = np.linalg.norm(h_i)**2
        interference = sum(p[j] * h_i_norm for j in sorted_users[i+1:])
        sinr = (p[i] * h_i_norm) / (interference + sigma2)
        rates.append(B * np.log2(1 + sinr))
    
    return sum(rates)  # Sum rate
print(f"NOMA System Sum Rate: {calculate_NOMA_rate() / 1e6:.2f} Mbps")