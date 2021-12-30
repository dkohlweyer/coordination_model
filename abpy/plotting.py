import matplotlib.pyplot as plt

from abpy import persistence
import numpy


class TimeSeriesPlot:
    def __init__(self):
        self.series = []
        self.labels = {}
        pass

    def add_series(self, datacollector, label=None):
        self.series.append(datacollector)
        if label is None:
            label = datacollector.name

        self.labels[datacollector] = label

    def save_plot(self, file, legend=True):
        persistence.ensure_dir(file)
        for dc in self.series:
            for key in dc.data.keys():
                plt.plot(dc.data[key], label=self.labels[dc])
        if legend:
            plt.legend(loc='best')

        plt.savefig(file)
        plt.clf()


class XYPlot:
    def __init__(self):
        self.points = {}

    def add_data_point(self, x, y, id=0):
        if id not in self.points:
            self.points[id]=[]
        self.points[id].append([x,y])

    def save_plot(self, file, legend=True):

        persistence.ensure_dir(file)

        for id in self.points:
            plt.plot(*zip(*self.points[id]), label=id)

        if legend:
            plt.legend(loc='best')

        plt.savefig(file)
        plt.clf()


class HistogramPlot:
    def __init__(self, bins, color='black'):
        self.bins = bins
        self.data = []
        self.color = color

    def add_data(self, datacollector, i=0):
        for id in datacollector.data:
            self.data.append(datacollector.data[id][i])

    def save_plot(self, file):

        persistence.ensure_dir(file)

        hist, bins = numpy.histogram(self.data, bins=self.bins)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width,color=self.color)
        plt.savefig(file)
        plt.clf()



