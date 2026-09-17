import os
import sys
import numpy as np

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PEANUTS_PATH = PROJECT_ROOT / "external" / "PEANUTS"


sys.path.append(str(PEANUTS_PATH))

print(PROJECT_ROOT, PEANUTS_PATH)

from peanuts.pmns import PMNS
from peanuts.solar import SolarModel, solar_flux_mass
from peanuts.earth import EarthDensity
from peanuts.evolutor import FullEvolutor


class SurvivalProbablity:

    def __init__(self, fraction='8B', depth=1e3):

        self.depth = depth

        self.fraction = fraction
        
        # Initialize th12 and DeltamSq21 without values
        self.th12 = None
        self.th13 = np.radians(8.62)
        self.th23 = np.radians(43.27)

        self.d = 3

        self.DeltamSq3l = 2.51e-3
        self.DeltamSq21 = None

        solar_model, self.earth_density = self._setup_solar_earth_model()
        self.radius_profile = solar_model.radius()
        self.density_profile = solar_model.density()
        self.flux_distributin = solar_model.fraction(self.fraction)

    def _setup_solar_earth_model(self):
        
        solar_file = PEANUTS_PATH / "Data" / "bs2005agsopflux.csv"
        density_file = PEANUTS_PATH / "Data" / "Earth_Density.csv"

        earth_density = EarthDensity(density_file=str(density_file))
        solar_model = SolarModel(str(solar_file))

        return solar_model, earth_density
    
    def _MSW(self, eta, enu):

        pmns = PMNS(self.th12, self.th13, self.th23, self.d)
        
        mass_weights = np.zeros((enu.shape[0],3))
        U_evolved = np.zeros((eta.shape[0], enu.shape[0], 3))
        U_evolved_vac = np.zeros((eta.shape[0], enu.shape[0], 3))

        for i in range (len(enu)):
            mass_weights[i] = solar_flux_mass(self.th12, self.th13, self.DeltamSq21, self.DeltamSq3l, enu[i],
                                    self.radius_profile, self.density_profile, self.flux_distributin)
            U_evolved_vac[:,i,:] = np.square(np.abs((pmns.pmns)))[0,:]
            if enu[i]<2: #less than 2 MeV the Earth matter effect become negligible
                U_evolved[:,i,:] = U_evolved_vac[:,i,:] 
            else:
                for j, angle in enumerate (eta):
                    evol = FullEvolutor(self.earth_density, self.DeltamSq21, self.DeltamSq3l, pmns, enu[i],
                                        angle, self.depth, False)
                    U_evolved[j,i] = np.square(np.abs((evol @ pmns.pmns)))[0,:]
                
        return U_evolved, mass_weights, U_evolved_vac  



