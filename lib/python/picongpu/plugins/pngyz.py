"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

import os
from scipy import misc
import collections

SPECIES_LONG_NAMES = {
    'e': 'Electrons'
}


class PNG(object):
    """
    Data reader for the PNG plugin.
    """

    def __init__(self, run_directory):
        """
        Parameters
        ----------
        run_directory: string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        if run_directory is None:
            raise ValueError('The run_directory parameter can not be None!')

        self.run_directory = run_directory
        self.data_file_prefix = "{0}_png_{1}_{2}_{3}"
        self.data_file_suffix = ".png"

    def get_data_path(self, species=None, species_filter="all",
                      axisyz=None, slice_point=None, iteration=None):
        """
        Return the path to the underlying data file.

        Parameters
        ----------
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)
        axis: string
            the coordinate system axis labels (e.g. 'yx' or 'yz')
        slice_point: float
            relative offset in the third axis not given in the axis argument.\
            Should be between 0 and 1.
            If not given, then it is determined from the (alphabetically)\
            first filename in the directory
        iteration: int
            The iteration at which to read the data.

        Returns
        -------
        A string with a path to a directory or to a specific file.
        If iteration is None, only the path to the directory where the png\
        files are located is returned. Otherwise the path to the png file that
        matches the requested iteration number is returned.
        """
        if species is None:
            raise ValueError("The species parameter can not be None!")
        if species_filter is None:
            raise ValueError('The species_filter parameter can not be None!')
        if axisyz is None:
            raise ValueError('The axis parameter can not be None!')

        dir_name = "png" + SPECIES_LONG_NAMES.get(species) + axisyz.upper()

        output_dir = os.path.join(
            self.run_directory,
            "simOutput",
            dir_name
        )

        if not os.path.isdir(output_dir):
            raise IOError('The simOutput/{0} directory does not '
                          'exist inside path:\n  {1}\n'
                          'Did you set the proper path to the run directory?\n'
                          'Did the simulation already run?'
                          .format(dir_name, self.run_directory))

        if iteration is None:
            return output_dir
        else:
            if slice_point is None:
                # determine slice point manually as the slice point of the
                # first png file in alphabetical order
                slice_point = [
                    f.split("_")[3] for f in sorted(os.listdir(output_dir)) if
                    f.endswith(".png")][0]
                slice_point = float(slice_point)

            data_file_name = self.data_file_prefix.format(
                species,
                axisyz,
                str(slice_point),
                '{:0>#6d}'.format(iteration)  # leading zeros for iter
            ) + self.data_file_suffix

            data_file_path = os.path.join(
                output_dir,
                data_file_name
            )

            if not os.path.isfile(data_file_path):
                raise IOError('The file {} does not exist.\n'
                              'Did the simulation already run?\n'
                              'Is there a png for this iteration?'
                              .format(data_file_path))

            return data_file_path

    def get_iterations(self, species, species_filter='all', axisyz=None,
                       slice_point=None):
        """
        Return an array of iterations with available png files.

        Parameters
        ----------
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)
        axis: string
            the coordinate system axis labels (e.g. 'yx' or 'yz')
        slice_point: float
            relative offset in the third axis not given in the axis argument.\
            Should be between 0 and 1
        iteration: int
            The iteration at which to read the data.

        Returns
        -------
        A sorted list of unsigned integers.
        """
        # get the available png files in the directory
        png_path = self.get_data_path(
            species, species_filter, axisyz, slice_point)
        # list of all png image file names
        png_files = [f for f in os.listdir(png_path) if f.endswith(".png")]

        # split iteration number from the filenames
        iters = [int(f.split("_")[4].split(".")[0]) for f in png_files]

        return sorted(iters)

    def get(self, species, species_filter='all', iteration=None,
            axisyz=None, slice_point=None, **kwargs):
        """
        Get an array representation of a PNG file.

        Parameters
        ----------
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)
        axis: string
            the coordinate system axis labels (e.g. 'yx' or 'yz')
        slice_point: float
            relative offset in the third axis not given in the axis argument.\
            Should be between 0 and 1
        iteration: int or list of ints
            The iteration at which to read the data.
            if set to 'None', then return images for all available iterations

        Returns
        -------
        A dictionary mapping iteration number to numpy array representation of
        the corresponding png file if multiple iterations were requested.
        Otherwise a single nump array representation for the requested\
        iteration.
        """
        available_iterations = self.get_iterations(
            species, species_filter, axisyz, slice_point)

        if iteration is not None:
            if not isinstance(iteration, collections.Iterable):
                iteration = [iteration]
            # verify requested iterations exist
            if not set(iteration).issubset(available_iterations):
                raise IndexError('Iteration {} is not available!\n'
                                 'List of available iterations: \n'
                                 '{}'.format(iteration, available_iterations))
        else:
            # iteration is None, so we use all available data
            iteration = available_iterations

        imgs = {it: misc.imread(
            self.get_data_path(species, species_filter, axisyz,
                               slice_point, it)) for it in iteration}
        if len(iteration) == 1:
            return imgs[iteration[0]]
        else:
            return imgs
