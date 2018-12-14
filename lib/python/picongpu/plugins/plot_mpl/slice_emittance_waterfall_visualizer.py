"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sophie Rudat, Sebastian Starke
License: GPLv3+
"""

from picongpu.plugins.data import EmittanceData
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


class Visualizer(BaseVisualizer):
    """
    Class for creation of waterfall plot with the slice emittance value
    for each y_slice (x-axis) and iteration (y-axis).
    """

    def __init__(self, run_directories=None, ax=None):
        """
        Parameters
        ----------
        run_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        super().__init__(EmittanceData, run_directories, ax)
        self.cbar = None

    def _create_plt_obj(self, idx):
        """
        Implementation of base class function.
        Turns 'self.plt_obj' into a matplotlib.pyplot.plot object.
        """
        slice_emit, y_slices, all_iterations, dt = self.data[idx]
        np_data = np.zeros((len(y_slices), len(all_iterations)))
        for index, ts in enumerate(all_iterations):
            np_data[:, index] = slice_emit[ts][1:]
        ps = 1.e12  # for conversion from s to ps
        max_iter = max(all_iterations * dt * ps)
        # np_data.T * 1.e6 converts emittance to pi mm mrad,
        # y_slices * 1.e6 converts y slice position to micrometer
        self.plt_obj[idx] = self.ax.imshow(np_data.T * 1.e6, aspect="auto",
                                 norm=LogNorm(), origin="lower",
                                 vmin=1e-1, vmax=1e2,
                                 extent=(0, max(y_slices*1.e6),
                                         0, max_iter))
        if self.cur_iteration:
            self.plt_lin = self.ax.axhline(self.cur_iteration * dt * ps,
                                      color='#FF6600')
        self.cbar = plt.colorbar(self.plt_obj[idx], ax=self.ax)
        self.cbar.set_label(r'emittance [$\mathrm{\pi mm mrad}$]')

    def _update_plt_obj(self, idx):
        """
        Implementation of base class function.
        """
        slice_emit, y_slices, all_iterations, dt = self.data[idx]
        np_data = np.zeros((len(y_slices), len(all_iterations)))
        for index, ts in enumerate(all_iterations):
            np_data[:, index] = slice_emit[ts][1:]
        # np_data.T*1.e6 for conversion of emittance to pi mm mrad
        self.plt_obj[idx].set_data(np_data.T*1.e6)
        if self.plt_lin:
            self.plt_lin.remove()
        ps = 1.e12  # for conversion from s to ps
        if self.cur_iteration:
            self.plt_lin = self.ax.axhline(self.cur_iteration * dt * ps,
                                           color='#FF6600')
        self.cbar.update_normal(self.plt_obj[idx])

    def visualize(self, ax=None, **kwargs):
        """
        Creates a semilogy plot on the provided axes object for
        the data of the given iteration using matpotlib.

        Parameters
        ----------
        iteration: int
            the iteration number for which data will be plotted.
        ax: matplotlib axes object
            the part of the figure where this plot will be shown.
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
        super().visualize(**kwargs)

    def adjust_plot(self, **kwargs):
        species = kwargs['species']
        species_filter = kwargs.get('species_filter', 'all')

        self.ax.set_xlabel(r'y-slice [$\mathrm{\mu m}$]')
        self.ax.set_ylabel('time [ps]')
        self.ax.set_title('slice emittance for species ' +
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
        Visualizer(path).visualize(ax, iteration=iteration, species=species,
                                   species_filter=filtr)
        plt.show()

    main()
