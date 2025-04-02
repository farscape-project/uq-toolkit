"""Use POD and Sklearn-GaussianProcessRegression to reconstruct a field"""
import numpy as np
import pyssam
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from skops.io import dump, get_untrusted_types, load
import xgboost as xgb

class Reconstructor:
  def __init__(self, xgb_fname, pod_coefs_fname, model_type="gp"):
    if model_type == "gp":
        self._load_gp_regressor(xgb_fname)
    else:
        raise NotImplementedError(f"model_type={model_type} not supported")
    self._load_pyssam(pod_coefs_fname)

  def _load_gp_regressor(self, gp_fname):
    """ 
    Read file with Gaussian process kernels etc.
    Prerequisite is that `train_surrogate.py` has been run already.

    Parameters
    ----------
    gp_fname : str
        /path/to/gp_model.skops
    """
    unknown_types = get_untrusted_types(file=gp_fname)
    self.surrogate_model = load(gp_fname, trusted=unknown_types)

  def _load_xgb_regressor(self, xgb_fname):
    """ 
    Read file with XGBoost config and weights.
    Prerequisite is that `train_xgb.py` has been run already.

    Parameters
    ----------
    xgb_fname : str
        /path/to/xgb_model.bin
    """
    self.xgb_model = xgb.Booster()
    self.xgb_model.load_model(xgb_fname)

  def _load_pyssam(self, pyssam_fname):
    """ 
    Create a dummy pyssam object, and read file with POD information.
    Prerequisite is that `find_pod_modes.py` has been run already.

    Parameters
    ----------
    pyssam_fname : str
        /path/to/pod_data.npz
    """
    # TODO: Implement such that object can be used with no dataset
    # i.e. train offline and use obj at inference time
    # make pyssam.morph_model a staticmethod
    self.sam_obj = pyssam.SAM(
      np.random.normal(size=(3, 3))
    )  # create dummy sam_obj
    npzfile = np.load(pyssam_fname)
    self.mean_dataset_columnvector = npzfile["mean"]
    self.pca_model_components = npzfile["pca_components"]
    self.sam_obj.std = npzfile["pca_std"]

  def reconstruct_with_gpr(
    self, t, param_list, reduction=None, num_modes=2
  ):
    """
    Reconstruct a field using POD pre-defined modes, and mode coefficients
    determined by an xgboost-regression model.
    The mode coefficients are based on time, t, and some parameters.

    Parameters
    ----------
    t : float
        physical time value to reconstruct
    param_list : list
        parameters needed for doing inference on the xgboost model
        ordering must be same as defined during training.
    reduction : function or None
        optional operation to apply to data such as np.max, np.mean, when only 
        a single value is needed
    num_modes : int
        number of POD modes to use in reconstruction
    
    Returns
    -------
    recon_field : array_lie 
        Reconstructed field values (or, optionally reduced to scalar)
    """
    feat_mat = np.array([[t, *param_list]])
    pod_coefs = np.array(self.surrogate_model.predict(feat_mat)).squeeze()

    # fix for when num_modes > pod_coefs
    num_modes = min(len(pod_coefs), num_modes)

    recon_field = self.sam_obj.morph_model(
      self.mean_dataset_columnvector,
      self.pca_model_components,
      pod_coefs[:num_modes],
      num_modes=num_modes,
    )
    if reduction is not None:
      return reduction(recon_field)
    else:
      return recon_field
