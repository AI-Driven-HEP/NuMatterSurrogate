import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physics import SurvivalProbablity

# ============================================================
# Parameter ranges
# ============================================================

#based on lates results from http://www.nu-fit.org/

TH12_MIN = 30.0       # degrees
TH12_MAX = 40.0       # degrees
N_TH     = 21

DM21_MIN = 6.0e-5     # eV^2
DM21_MAX = 9.0e-5     # eV^2
N_DM     = 21

ETA_MIN = 0.0         # degrees
ETA_MAX = 95.0       # degrees
N_ETA   = 192

E_MIN = 5.0           # MeV
E_MAX = 20.0          # MeV
N_E   = 192

# ============================================================
# Sample generator
# ============================================================

def generate_samples():
    """
    Generate parameter samples for the matter-effect surrogate.

    Parameters
    ----------
    n_parameter_samples : int
        Number of (theta12, dm21) parameter samples.

    seed : int
        Random seed.

    Returns
    -------
    samples : dict
        Dictionary containing the sampled parameters.
    """

    theta12 = np.linspace(
        TH12_MIN,
        TH12_MAX,
        N_TH)
    
    dm21 = np.linspace(
        DM21_MIN,
        DM21_MAX,
        N_DM)
    
    eta = np.linspace(
        ETA_MIN,
        ETA_MAX,
        N_ETA)

    energy  = np.logspace(
        np.log10(E_MIN),
        np.log10(E_MAX),
        N_E)

    n_parameter_samples = int(N_TH * N_DM)

    model = SurvivalProbablity()
    U_Evol = np.zeros((n_parameter_samples, N_ETA, N_E, 3))
    Mass_W = np.zeros((n_parameter_samples, N_E, 3))
    param  = np.zeros((n_parameter_samples, 2))
    k = 0
    for i in range(N_TH):
        model.th12 = np.radians(theta12[i]) 
        for j in range (N_DM):
            model.DeltamSq21 = dm21[j]
            U_Evol[k], Mass_W[k], _  = model._MSW(np.radians(eta), energy)
            param[k,0] = theta12[i]
            param[k,1] = dm21[j]
            k = k + 1


    return {
        "param": param,
        "eta": eta,
        "energy": energy,
        "U_Evol" : U_Evol,
        "mass_weights" : Mass_W,
    }


# ============================================================
# Save
# ============================================================

def save_samples(samples, filename):

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        filename,
        param=samples["param"],
        eta=samples["eta"],
        energy=samples["energy"],
        U_Evol=samples["U_Evol"],
        Mass_W=samples["mass_weights"]
    )

    print(f"Saved samples to: {filename}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    samples = generate_samples()

    save_samples(
        samples,
        ROOT / "data" / "parameter_samples.npz"
    )
