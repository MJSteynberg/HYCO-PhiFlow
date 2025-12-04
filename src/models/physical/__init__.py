# src/models/physical/__init__.py

# This file makes your model classes importable

# Import the base class so it can be accessed if needed
from .base import PhysicalModel


# You can add other models here later
from .burgers import BurgersModel
from .inviscid_burgers import InviscidBurgersModel
from .advection import AdvectionModel
from .navier_stokes import NavierStokesModel