from collections import defaultdict
import logging
from typing import List, Dict, Optional

import numpy as np

from bolos.process import Process


class Target:
    """ A class to contain all information related to one target. """

    def __init__(self, name: str) -> None:
        """ Initializes an instance of target named name."""
        self.name = name
        self.mass_ratio: Optional[float] = None
        self.density = 0.0

        # Lists of all processes pertaining this target
        self.elastic: List[Process] = []
        self.effective: List[Process] = []
        self.attachment: List[Process] = []
        self.ionization: List[Process] = []
        self.excitation: List[Process] = []
        self.weighted_elastic: List[Process] = []

        self.kind = {'ELASTIC': self.elastic,
                     'EFFECTIVE': self.effective,
                     'MOMENTUM': self.effective,
                     'ATTACHMENT': self.attachment,
                     'IONIZATION': self.ionization,
                     'EXCITATION': self.excitation,
                     'WEIGHTED_ELASTIC': self.weighted_elastic}

        self.by_product: Dict[str, List[Process]] = defaultdict(list)

        logging.debug("Target %s created.", str(self))

    def add_process(self, process: Process) -> None:
        kind: List[Process] = self.kind[process.kind]
        kind.append(process)

        if process.mass_ratio is not None:
            logging.debug(
                f"Mass ratio (={process.mass_ratio}) for {str(self)}")

            if (self.mass_ratio is not None
                    and self.mass_ratio != process.mass_ratio):
                raise ValueError(
                    f"More than one mass ratio for target '{self.name}'")

            self.mass_ratio = process.mass_ratio

        process.target = self

        if process.product is None and process.kind != 'EFFECTIVE' and process.kind != 'ELASTIC':
            raise ValueError(
                f"Attribute `product` has not been set for process `{process}`")
        self.by_product[process.product].append(process)

        logging.debug(f"Process{str(process)} added to target {str(self)}")

    def ensure_elastic(self) -> None:
        """ Makes sure that the process has an elastic cross-section.
        If the user has specified an effective cross-section, we remove
        all the other cross-sections from it. """
        if self.elastic and self.effective:
            raise ValueError(f"In target '{self}': EFFECTIVE/MOMENTUM and ELASTIC"
                             "cross-sections are incompatible.")

        if self.elastic:
            return

        if len(self.effective) > 1:
            raise ValueError(f"In target '{self}': Can't handle more that 1 "
                             "EFFECTIVE/MOMENTUM for a given target")

        if not self.effective:
            logging.warning(f"Target '{str(self)}' has no ELASTIC or EFFECTIVE "
                            "cross sections")
            return

        new_data = self.effective[0].data.copy()
        for p in self.inelastic:
            new_data[:, 1] -= p.interp(new_data[:, 0])

        if np.amin(new_data[:, 1]) < 0:
            logging.warning('After substracting INELASTIC from EFFECTIVE,  target %s has negative cross-section.',
                            self.name)
            logging.warning('Setting as max(0, ...)')
            new_data[:, 1] = np.where(new_data[:, 1] > 0, new_data[:, 1], 0)

        newelastic = Process(target=self.name,
                             kind='ELASTIC',
                             data=new_data,
                             comment="Calculated from EFFECTIVE cross sections",
                             mass_ratio=self.effective[0].mass_ratio)

        logging.debug("EFFECTIVE -> ELASTIC for target %s", str(self))
        self.add_process(newelastic)

        # Remove the EFFECTIVE processes.
        self.effective = []

    @property
    def inelastic(self) -> List[Process]:
        """ An useful abbreviation. """
        return self.attachment + self.ionization + self.excitation

    @property
    def everything(self) -> List[Process]:
        """ A list with ALL processes.  We do not use all as a name
        to avoid confusion with the python function."""
        return (self.elastic + self.attachment
                + self.ionization + self.excitation)

    def __repr__(self) -> str:
        return f"Target({repr(self.name)})"

    def __str__(self) -> str:
        return self.name
