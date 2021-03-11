#adapted from https://github.com/pyimreg/python-register

import numpy as np
import scipy.ndimage as nd

class Coordinates(object):
    """
    Container for grid coordinates.
    Attributes
    ----------
    domain : nd-array
        Domain of the coordinate system.
    tensor : nd-array
        Grid coordinates.
    homogenous : nd-array
        `Homogenous` coordinate system representation of grid coordinates.
    """

    def __init__(self, domain, spacing=None):

        self.spacing = 1.0 if not spacing else spacing
        self.domain = domain
        self.tensor = np.mgrid[0.:domain[1], 0.:domain[3]]

        self.homogenous = np.zeros((3,self.tensor[0].size))
        self.homogenous[0] = self.tensor[1].flatten()
        self.homogenous[1] = self.tensor[0].flatten()
        self.homogenous[2] = 1.0


class RegisterData(object):
    """
    Container for registration data.
    Attributes
    ----------
    data : nd-array
        The image registration image values.
    features : dictionary, optional
        A mapping of unique ids to registration features.
    """

    def __init__(self, data, features=None, spacing=1.0):

        self.data = data.astype(np.double)
        self.coords = Coordinates(
                [0, data.shape[0], 0, data.shape[1]],
                spacing=spacing
                )
        self.features = features