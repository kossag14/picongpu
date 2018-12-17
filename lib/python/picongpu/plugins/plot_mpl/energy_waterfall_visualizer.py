"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sophie Rudat, Sebastian Starke
License: GPLv3+
"""

from picongpu.plugins.data import EnergyHistogramData
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


class Visualizer(BaseVisualizer):
    """
    Class for creation of histogram plots on a logscaled y-axis.
    """

    def __init__(self, run_directories=None, ax=None):
        """
        Parameters
        ----------
        run_directory : list of tuples of length 2
            or single tuple of length 2.
            Each tuple is of the following form (sim_name, sim_path)
            and consists of strings.
            sim_name is a short string used e.g. in plot legends.
            sim_path leads to the run directory of PIConGPU
            (the path before ``simOutput/``).
            If None, the user is responsible for providing run_directories
            later on via set_run_directories() before calling visualize().
        ax: matplotlib.axes
        """
        super().__init__(EnergyHistogramData, run_directories, ax)
        self.cbar = None
        self.plt_lin = None  # plot line at current itteration
        self.cur_itteration = None

    def _create_plt_obj(self, idx):
        """
        Implementation of base class function.
        Turns 'self.plt_obj' into a matplotlib.pyplot.plot object.
        """
        counts, bins, all_iterations, dt = self.data[idx]
        np_data = np.zeros((len(bins), len(all_iterations)))
        for index, ts in enumerate(all_iterations):
            np_data[:, index] = counts[index]
        ps = 1.e12  # for conversion from s to ps
        max_iter = max(all_iterations * dt * ps)
        self.plt_obj[idx] = self.ax.imshow(np_data, aspect="auto",
                                           norm=LogNorm(),
                                           origin="lower",
                                           extent=(0, max_iter,
                                                   0, max(bins*1.e-3)))
        if self.cur_iteration:
            self.plt_lin = self.ax.axvline(self.cur_iteration * dt * ps,
                                           color='#FF6600')
        self.cbar = plt.colorbar(self.plt_obj[idx], ax=self.ax)
        self.cbar.set_label(r'Count')

    def _update_plt_obj(self, idx):
        """
        Implementation of base class function.
        """
        counts, bins, all_iterations, dt = self.data[idx]
        np_data = np.zeros((len(bins), len(all_iterations)))
        for index, ts in enumerate(all_iterations):
            np_data[:, index] = counts[index]
        self.plt_obj[idx].set_data(np_data)
        if self.plt_lin:
            self.plt_lin.remove()
        ps = 1.e12  # for conversion from s to ps
        if self.cur_iteration:
            self.plt_lin = self.ax.axvline(self.cur_iteration * dt * ps,
                                           color='#FF6600')
        self.cbar.update_normal(self.plt_obj[idx])

    def visualize(self, **kwargs):
        """
        Creates a semilogy plot on the provided axes object for
        the data of the given iteration using matpotlib.

        Parameters
        ----------
        kwargs: dictionary with further keyword arguments, valid are:
            species: string
                short name of the particle species, e.g. 'e' for electrons
                (defined in ``speciesDefinition.param``)
            iteration: int
                number of the iteration
            species_filter: string
                name of the particle species filter, default is 'all'
                (defined in ``particleFilters.param``)

        """
        self.cur_iteration = kwargs.get('iteration')
        kwargs['iteration'] = None
        # this already throws error if no species or iteration in kwargs
        super().visualize(**kwargs)

    def adjust_plot(self, **kwargs):
        species = kwargs['species']
        species_filter = kwargs.get('species_filter', 'all')

        self.ax.set_xlabel('time [ps]')
        self.ax.set_ylabel('Energy [MeV]')
        self.ax.set_title('Energy Histogram for species ' +
                          species + ', filter = ' + species_filter)

    def clear_cbar(self):
        """Clear colorbar if present."""
        if self.cbar is not None:
            self.cbar.remove()


if __name__ == '__main__':

    def main():
        import sys
        import getopt

        def usage():
            print("usage:")
            print(
                "python", sys.argv[0],
                "-p <path to run_directory>"
                " -s <particle species> -f <species_filter> -i <iteration>")

        path = None
        iteration = None
        species = None
        filtr = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "hp:i:s:f:", [
                "help", "path", "iteration", "species", "filter"])
        except getopt.GetoptError as err:
            print(err)
            usage()
            sys.exit(2)

        for opt, arg in opts:
            if opt in ["-h", "--help"]:
                usage()
                sys.exit()
            elif opt in ["-p", "--path"]:
                path = arg
            elif opt in ["-i", "--iteration"]:
                iteration = int(arg)
            elif opt in ["-s", "--species"]:
                species = arg
            elif opt in ["-f", "--filter"]:
                filtr = arg

        # check that we got all args that we need
        if path is None:
            print("Path to 'run' directory have to be provided!")
            usage()
            sys.exit(2)
        if species is None:
            species = 'e'
            print("Particle species was not given, will use", species)
        if filtr is None:
            filtr = 'all'
            print("Species filter was not given, will use", filtr)

        fig, ax = plt.subplots(1, 1)
        Visualizer(path, ax).visualize(iteration=iteration, species=species,
                                   species_filter=filtr)
        plt.show()

    main()
