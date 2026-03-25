import numpy as np
import random
import pandas as pd

class RDFResult:
    def __init__(self, box_size, r, rdf, rdf_std, rdf_err):
        self.box_size = box_size
        self.r = r
        self.rdf = rdf
        self.rdf_std = rdf_std
        self.rdf_err_up = rdf_err

    def __repr__(self):
        return (f"RDFResult(box_size={self.box_size}, r={self.r}, rdf={self.rdf}, rdf_std={self.rdf_std}, rdf_err_up={self.rdf_err_up}")


def df_to_dict(df):
    rdfs = {}
    column_names = df.columns
    for column_name in column_names:
    # Extract r values and rdf values
        r_values = df[column_name][0]  # Assuming r values are in the first row (index 0)
        rdf_values = df[column_name][1]  # Assuming rdf values are in the second row (index 1)
        
        # Store results in the dictionary
        rdfs[column_name] = [r_values, rdf_values]
    
    return rdfs
        

def calc_errors(rdfs, name):
    rdf_errors_ind = {}
    rdf_errors_all = {}
    rdf_sub_dict = {}
    rdf_dist = {}
    for column_name in rdfs.keys():  # This will cover run_1 to run_10
        # Extract r values and rdf values
        r_values = rdfs[column_name][0]  # Assuming r values are in the first row (index 0)
        rdf_values = rdfs[column_name][1]  # Assuming rdf values are in the second row (index 1)

        for rij, rdfij in zip(r_values, rdf_values):
            if rij not in rdf_dist:
                rdf_dist[rij] = []  # Initialize if rij not present
            rdf_dist[rij].append(rdfij)  # Append rdfij to the accumulated list
            if rij not in rdf_sub_dict:
                rdf_sub_dict[rij] = []  # Initialize if rij not present
            rdf_sub_dict[rij].append(rdfij)  # Append rdfij to the accumulated list

        rdf_err=[]
        rdf_std = []
        rdf_means = []
        for key, val in rdf_sub_dict.items():
            mean = np.mean(val)
            std_dev = np.std(val)
            n = len(val)
            std_error = std_dev / np.sqrt(n)
            rdf_err.append(mean+std_dev)
            rdf_std.append(std_dev)
            rdf_means.append(mean)
        # Create an RDFResult object for the current run
        rdf_result = RDFResult(column_name, list(rdf_sub_dict.keys()), rdf_means, rdf_std, rdf_err)
        rdf_errors_ind.append(rdf_result)

    rdf_err=[]
    rdf_std = []
    rdf_means = []
    for key, val in rdf_dist.items():
        mean = np.mean(val)
        std_dev = np.std(val)
        n = len(val)
        std_error = std_dev / np.sqrt(n)
        rdf_err.append(mean+std_dev)
        rdf_std.append(std_dev)
        rdf_means.append(mean)
    # Create an RDFResult object for the current run
    rdf_errors_all = RDFResult(name, list(rdf_dist.keys()), rdf_means, rdf_std, rdf_err)
    return rdf_errors_ind, rdf_errors_all


def calc_errors_through_runs(df, box_size):
    rdfs = {}
    rdf_sub_dict = {}
    for i in range(1, 11):  # This will cover run_1 to run_10
        column_name = f'run_{i}'
        if column_name in df.columns:
            # Extract r values and rdf values
            r_values = df[column_name][0]  # Assuming r values are in the first row (index 0)
            rdf_values = df[column_name][1]  # Assuming rdf values are in the second row (index 1)
            
            # Store results in the dictionary
            rdfs[column_name] = [r_values, rdf_values]

            for rij, rdfij in zip(r_values, rdf_values):
                if rij in rdf_sub_dict.keys():
                    #print("rij", rij)
                    rdf_sub_dict[rij].append(rdfij)
                else:
                    list_1 = []
                    rdf_sub_dict[rij] = list_1
                    rdf_sub_dict[rij].append(rdfij)            
    
    rdf_err=[]
    rdf_std = []
    rdf_means = []
    for key, val in rdf_sub_dict.items():
        mean = np.mean(val)
        std_dev = np.std(val)
        n = len(val)
        std_error = std_dev / np.sqrt(n)
        rdf_err.append(mean+std_dev)
        rdf_std.append(std_dev)
        rdf_means.append(mean)
    # Create an RDFResult object for the current run
    rdf_errors = RDFResult(box_size, list(rdf_sub_dict.keys()), rdf_means, rdf_std, rdf_err)
    return rdfs, rdf_errors


def calc_b22(rdfs, last):
    B22 = {}
    B22_E = {}
    B22_runs = {}
    bootstrap_runs = {}
    B22_all = []
    bootstrap_all = []
    for column_name in rdfs.keys():  # This will cover run_1 to run_10
        # Extract r values and rdf values
        r_values = rdfs[column_name][0]  # Assuming r values are in the first row (index 0)
        rdf_values = rdfs[column_name][1]  # Assuming rdf values are in the second row (index 1)            

        B22_value = calculate_b2(r_values, np.array(rdf_values), last)
        B22_runs[column_name].append(B22_value)
        B22_all.append(B22_value)

        # Perform bootstrapping (1000 resamples)
        for _ in range(1000):
            picked = random.choices(B22_runs[column_name], k=len(B22_runs[column_name]))
            bootstrap_runs[column_name].append(np.mean(picked))

        # Calculate the mean and error for B22
        B22[column_name] = np.mean(B22_runs[column_name])
        B22_E[column_name] = np.std(bootstrap_runs[column_name]) / np.sqrt(len(B22_runs[column_name]))  # Standard error of the mean
        
    for _ in range(1000):
        picked = random.choices(B22_all, k=len(B22_all))
        bootstrap_all.append(np.mean(picked))

    return B22, B22_E, np.mean(B22_all), np.std(bootstrap_all) / np.sqrt(len(B22_all))


def calc_delta_N_ij(r, g_r, N_j, V):
    """
    Calculate the excess number of protein j around protein i (ΔN_ij(r)) using Eq. 3.10.
    
    Parameters:
    - r: numpy array of r values
    - g_r: numpy array of uncorrected g(r) values
    - N_j: number of type j chains (assumed to be 1 here)
    - V: volume of the simulation box (in the same units as r^3)
    
    Returns:
    - delta_N_ij: numpy array of ΔN_ij(r)
    """

    # Ensure r and g_r are numpy arrays
    r = np.asarray(r, dtype=np.float64)
    g_r = np.asarray(g_r, dtype=np.float64)

    # Perform the integration using cumulative trapezoidal rule
    integrand = 4 * np.pi * r**2 * (g_r - 1)  # [4πr²(g(r) - 1)]
    delta_N_ij = (N_j / V) * np.trapz(integrand, r)
    
    return delta_N_ij


def calc_corr_rdf_ganguly(r, g_r, N_j, L, delta_ij=0):
    """
    Calculate the corrected RDF values (g_ij^correct(r)) using Eq. 3.9.
    
    Parameters:
    - r: numpy array of r values
    - g_r: numpy array of uncorrected g(r) values
    - N_j: number of type j chains (assumed to be 1 here)
    - V: volume of the simulation box (in the same units as r^3)
    - delta_N_ij: numpy array of ΔN_ij(r)
    - delta_ij: Kronecker delta, default is 0 as described
    
    Returns:
    - g_corrected: numpy array of corrected g_ij(r)
    """

    # Ensure r and g_r are numpy arrays
    r = np.asarray(r, dtype=np.float64)
    g_r = np.asarray(g_r, dtype=np.float64)

    V = L**3

    delta_N_ij = calc_delta_N_ij(r, g_r, N_j, V)

    # Calculate the correction factor
    correction_factor = N_j * (1 - (4/3) * np.pi * r**3 / V)
    
    # Calculate the corrected g(r)
    g_corrected = g_r * (correction_factor / (correction_factor - delta_N_ij - delta_ij))
    
    return g_corrected


def calculate_b2(r, rdf, last):
    # Sum all values of (rdf - 1)
    
    mask = r <= last
    #print(mask)
    r_integral = r[mask]
    rdf_integral = rdf[mask]
    integrand = (rdf_integral - 1) * r_integral**2
    B22 = -2 * np.pi * np.sum(integrand)
    return B22 

# Constants
NA = 6.02214076e23  # Avogadro's number

def calculate_cij(r, g_r, r_star, delta):
    """Calculate c_ij (RDF tail average) over a window from (r_star - delta) to r_star."""
    mask = (r >= (r_star - delta)) & (r <= r_star)
    g_r_window = g_r[mask]
    r_window = r[mask]
    
    # Numerical integration to get the average value
    cij = (np.trapz(g_r_window, r_window)) / delta
    return cij


def calculate_bij_hummer(r, g_r, r_star, delta):
    """Calculate second virial coefficient B_ij."""
    # Calculate c_ij
    cij = calculate_cij(r, g_r, r_star, delta)

    corr_rdf = g_r - cij + 1
    
    # Create mask to integrate from 0 to r_star
    mask = r <= r_star
    r_integral = r[mask]
    g_r_integral = g_r[mask]
    
    # Calculate the integrand: [g_ij(r) - c_ij] * r^2
    integrand = (g_r_integral - cij) * r_integral**2
    
    # Numerical integration over r from 0 to r_star
    bij = -2 * np.pi * np.sum(integrand)
    
    return bij, corr_rdf


def corrected_rdfs_b22(df, last, box_size=None, delta=2, rc_nm=4.0):
    rdfs = df_to_dict(df)
    rdfs_dict = {}
    b22s_dict = {}
    kds_dict = {}
    for column_name in rdfs.keys():  # This will cover run_1 to run_10
        # Extract r values and rdf values
        r_values = rdfs[column_name][0]  # Assuming r values are in the first row (index 0)
        rdf_values = rdfs[column_name][1]  # Assuming rdf values are in the second row (index 1)
        b22_corr1, rdf_corr1 = calculate_bij_hummer(r_values, rdf_values, last, delta)
        if box_size:    
            rdf_corr2 = calc_corr_rdf_ganguly(r_values, rdf_values, N_j=1, L=box_size)
        else:
            rdf_corr2 = calc_corr_rdf_ganguly(r_values, rdf_values, N_j=1, L=column_name)
        b22_corr2 = calculate_b2(r_values, rdf_corr2, last)
        b22 = calculate_b2(r_values, rdf_values, last)

        # Store results in the dictionary with sub-keys
        rdfs_dict[column_name] = {
            'rs': r_values,
            'rdfs': rdf_values,
            'rdfs_corr1': rdf_corr1,
            'rdfs_corr2': rdf_corr2
        }
        # Store results in the dictionary with sub-keys
        b22s_dict[column_name] = {
            'b22': b22,
            'b22_corr1': b22_corr1,
            'b22_corr2': b22_corr2
        }

        # --- NEW: Ka/Kd from g(r), 0→rc ---
        Ka_raw,   Kd_raw   = _kd_from_curve(r_values, rdf_values, rc_nm)
        Ka_c1,    Kd_c1    = _kd_from_curve(r_values, rdf_corr1, rc_nm)
        Ka_c2,    Kd_c2    = _kd_from_curve(r_values, rdf_corr2, rc_nm)

        kds_dict[column_name] = {
            'rc_nm':    rc_nm,
            'Ka':       Ka_raw,    'Kd':       Kd_raw,    'Kd_uM':   (Kd_raw*1e6 if np.isfinite(Kd_raw) else np.nan),
            'Ka_corr1': Ka_c1,     'Kd_corr1': Kd_c1,     'Kd_corr1_uM': (Kd_c1*1e6 if np.isfinite(Kd_c1) else np.nan),
            'Ka_corr2': Ka_c2,     'Kd_corr2': Kd_c2,     'Kd_corr2_uM': (Kd_c2*1e6 if np.isfinite(Kd_c2) else np.nan),
        }

    # Convert the nested dictionary to a DataFrame
    df_results = pd.DataFrame.from_dict(rdfs_dict, orient='index')
    
    return rdfs_dict, b22s_dict, kds_dict







import numpy as np, math
V0_NM3 = 1.660  # nm^3 at 1 M

def _Ka_from_gr(r, g, rc_nm):
    m = (r >= 0.0) & (r <= rc_nm)
    if m.sum() < 2:
        return np.nan
    I_nm3 = np.trapz(4.0 * math.pi * (r[m]**2) * g[m], r[m])
    return I_nm3 / V0_NM3  # M^-1

def _kd_from_curve(r, g, rc_nm):
    Ka = _Ka_from_gr(np.asarray(r), np.asarray(g), rc_nm)
    if not np.isfinite(Ka) or Ka <= 0:
        return np.nan, np.nan
    return Ka, 1.0/Ka
