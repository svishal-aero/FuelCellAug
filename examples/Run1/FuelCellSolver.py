import casadi
import pybamm
import pypemfc
import numpy as np
import matplotlib.pyplot as plt

import keras as K

class FuelCellSolver:

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def __init__(self):

    self.feature_name_list = [
      'Anode channel water vapor mole fraction', # Can drop this
      'Cathode channel temperature',
      'Cathode channel water vapor mole fraction', # Can drop this
      'X-averaged anode CL water content', # Some cases affected
      'X-averaged anode CL water vapor concentration',
      'X-averaged cathode CL water content', # Some cases affected
      'X-averaged cathode CL water vapor concentration',
      'X-averaged membrane water content',
    ]

    self.standard_params = pypemfc.standard_parameters
    
    param_fitted = pypemfc.parameters.Toyota_confidential.Toyota_fitted.get_parameters()
    
    self.data = pypemfc.parameters.Toyota_confidential.ValidationData()
    
    model = pypemfc.models.LinearizedModel({"channel dimension": 1,})

    param = pybamm.ParameterValues(param_fitted)
  
    param.update(
      {
          "Water diffusivity [m2.s-1]": self.water_diffusivity_Vetter2019,
          "Electro-osmotic drag coefficient": self.electro_osmotic_drag_Springer1991,
          "Saturation pressure [Pa]": self.saturation_pressure_Nam2003,
          "Reference diffusivity of water vapor in air [m2.s-1]": 0.36e-4*0.5958324939769479*pybamm.InputParameter("D_H2Oca", "channel"),
          "Desorption rate [m.s-1]": self.desorption_rate_Ge2005,
          "Adsorption rate [m.s-1]": self.adsorption_rate_Ge2005,
          "Ionic conductivity of membrane [S.m-1]": self.membrane_ionic_conductivity_Weber2004,
          "Oxygen reaction exchange-current density [A.m-2]"+"": self.oxygen_exchange_current_density_Toyota,
          "Membrane equilibrium water content anode": self.membrane_equilibrium_Springer1991,
          "Membrane equilibrium water content cathode": self.membrane_equilibrium_Springer1991,
          "Condensation rate [s-1]": self.condensation_rate_Vetter2019,
      }
    )
    
    param.process_model(model)

    spatial_vars = pypemfc.standard_spatial_vars
    N = 20
    var_pts = {
      spatial_vars.x_agdl: 2,
      spatial_vars.x_acl:  4,
      spatial_vars.x_mb:   4,
      spatial_vars.x_ccl:  4,
      spatial_vars.x_cgdl: 2,
      spatial_vars.y:      N,
    }
    submesh_types = model.default_submesh_types
    solver = pybamm.CasadiSolver(mode="safe", extra_options_setup={"max_num_steps":3000}) # Can also use mode="safe", "fast"
    self.sim = pybamm.Simulation(model, parameter_values=param, var_pts=var_pts, submesh_types=submesh_types, solver=solver)
    
    self.ones = np.ones((N,1))
    self.inputs_ones = {
      "D_lambda" : self.ones,
      "n_d"      : self.ones,
      "p_sat"    : self.ones,
      "D_H2Oca"  : self.ones,
      "k_ad"     : self.ones,
      "k_de"     : self.ones,
      "sigma_p"  : self.ones,
      "i_O2"     : self.ones,
      "lambda_eq": self.ones,
      "gamma_c"  : self.ones,
    }

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def inputs_lambdaeq(self, x):
    return {
      **self.inputs_ones,
      "gamma_c"   : 0.01 * self.ones,
      "lambda_eq" : x,
    }

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def calcObj(self, sol, outlets):
    lmbda_sol = sol["Lambda Distribution"].data[:,-1]
    lmbda_dat = outlets["Lambda Distribution [-]"].flatten()
    return np.linalg.norm(lmbda_sol - lmbda_dat)

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def calcFtrs(self, sol):
    ftrs = np.zeros((self.ones.size, len(self.feature_name_list)))
    for j in range(len(self.feature_name_list)):
      var = sol[self.feature_name_list[j]].data[:,-1]
      ftrs[:,j] = var[:]
    return ftrs

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def solveForCaseWithBeta(self, case_id=None, beta=None, nSolverIters=None):
    if beta is None: beta = self.ones
    if nSolverIters is None: nSolverIters = 100
    inlets = self.data.get_inlets(case_id)
    outlets = self.data.get_outlets(case_id)
    sol = self.sim.solve(np.linspace(0,nSolverIters), inputs={**inlets, **self.inputs_lambdaeq(beta)}) 
    obj = self.calcObj(sol, outlets)
    ftrs = self.calcFtrs(sol)
    currDist = np.zeros((20,2))
    currDist[:,0] = outlets["Current Distribution [A/cm2]"].flatten()
    currDist[:,1] = sol["Current Distribution [A/cm2]"].data[:,-1]
    np.savetxt("currDist.txt", currDist)
    return beta, obj, ftrs, sol['Lambda Distribution'].data[:,-1]

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def electro_osmotic_drag_Springer1991(self, lambda_mb):
    return 2.5 * lambda_mb / 22 / 10 * 1.0712613289919768 * pybamm.InputParameter("n_d", "channel")

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def saturation_pressure_Nam2003(self, T):
    p_sat = pybamm.InputParameter("p_sat", "channel")
    return (
        pybamm.exp(
            -5800 / T
            + 1.391
            - 0.04864 * T
            + 0.4176e-4 * (T ** 2)
            - 0.1445e-7 * (T ** 3)
            + 6.546 * pybamm.log(T)
        )
        * 0.6815757803578539 * p_sat
    )

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def water_diffusivity_Vetter2019(self, lambda_mb, T, eps_i):
    frac = (3.842 * lambda_mb ** 3 - 32.03 * lambda_mb ** 2 + 67.74 * lambda_mb) / (
        lambda_mb ** 3 - 2.115 * lambda_mb ** 2 - 33.013 * lambda_mb + 103.37
    )
    arrhenius = pybamm.exp(20e3 / self.standard_params.R * (1 / self.standard_params.T_ref - 1 / T))
    return (
        eps_i ** 1.5 * frac * 1e-10 * pybamm.InputParameter("D_lambda", "channel") * arrhenius
    )  # m2/s

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def adsorption_rate_Ge2005(self, lmbda, T):
    arrhenius = pybamm.exp(20e3 / self.standard_params.R * (1 / self.standard_params.T_ref - 1 / T))

    return 3.53e-5 * self.standard_params.f(lmbda) * arrhenius * pybamm.InputParameter("k_ad", "channel")

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def desorption_rate_Ge2005(self, lmbda, T):
    arrhenius = pybamm.exp(20e3 / self.standard_params.R * (1 / self.standard_params.T_ref - 1 / T))

    return 1.42e-4 * self.standard_params.f(lmbda) * arrhenius * 0.9283001909919966 * pybamm.InputParameter("k_de", "channel")

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def membrane_ionic_conductivity_Weber2004(self, lambda_mb, T, eps_i):
    f = self.standard_params.f(lambda_mb)
    arrhenius = pybamm.exp(15e3 / self.standard_params.R * (1 / self.standard_params.T_ref - 1 / T))
    # add residual conductivity to avoid singularity at zero
    # value from Alireza Goshtasbi's thesis
    sigma_p_res = 0.0025
    return (
        sigma_p_res
        + eps_i ** 1.5
        * 116
        * 1.679826859512183
        * pybamm.InputParameter("sigma_p", "channel")
        * pybamm.maximum(0, f - 0.06) ** 1.5
        * arrhenius
    )

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def oxygen_exchange_current_density_Toyota(self, c_O2, T):
    arrhenius = pybamm.exp(67e3 / self.standard_params.R * (1 / 353.15 - 1 / T))
    c_O2 = pybamm.maximum(c_O2, 0)

    return (
        3.4377e-3
        * 1.0287467814488682
        * pybamm.InputParameter("i_O2", "channel")
        * (c_O2 / self.standard_params.c_typ) ** 0.5012
        * arrhenius
    )

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def membrane_equilibrium_Springer1991(self, rh, T):
    return (0.043 + 17.81 * rh - 39.85 * rh ** 2 + 36.0 * rh ** 3) * pybamm.InputParameter("lambda_eq", "channel")

  #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  def condensation_rate_Vetter2019(self, T):
    return pybamm.sqrt(self.standard_params.R * T / (2 * self.standard_params.pi * self.standard_params.M_w)) * 6e-3 * self.standard_params.a_lg * pybamm.InputParameter("gamma_c", "channel")
