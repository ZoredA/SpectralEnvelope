import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

class PlotManager():
    def __init__(self, plotCount):
        self.off = False
        self.plots = {}

        self.plotCount = plotCount
        self.currentPlots = {}
        plt.ion()

    def addPlot(self, fig_title, axis_title, x_label, y_label, x, y, color='-r',
        x_limit=None, y_limit=None):
        if self.off: return
        if fig_title not in self.currentPlots:
            self.currentPlots[fig_title] = {
                'row':1,
                'column':1
            }

        fig = plt.figure(fig_title)
        ax  = fig.add_subplot(self.plotCount, 1, self.currentPlots[fig_title]['row'])
        #ax  = fig.add_subplot(211)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(axis_title)
        if x_limit is not None:
            ax.set_xlim(*x_limit)
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        ax.plot(x, y, color)

        self.currentPlots[fig_title]['row'] = self.currentPlots[fig_title]['row'] + 1


    def startPlot(self, fig_title, rows, columns, argsDict=None):
        if self.off: return
        figDict = {
            'num':fig_title,
            'edgecolor':'blue'
        }
        if argsDict is not None:
            argsDict = dict(figDict, **argsDict)
            fig, axis = plt.subplots(rows, columns, squeeze=False, **argsDict)
        else:
            fig, axis = plt.subplots(rows, columns, squeeze=False, **figDict)
        print(axis)
        self.plots[fig_title] = {
            'figure':fig,
            'axis':axis,
            'rows': rows,
            'columns': columns,
            'cur_row' : 0,
            'cur_column' : 0
        }

    def addPlotN(self, fig_title, axis_title, x_label, y_label, x, y, color='-r',
        x_limit=None, y_limit=None):
        if self.off: return
        if fig_title not in self.plots:
            raise Exception("Need to call startplot before addplot")
        info = self.plots[fig_title]
        if info['cur_row'] >= info['rows'] or info['cur_column'] >= info['columns']:
            print(info)
            raise Exception("Exceeded specified plot row/columns")

        ax = info['axis'][info['cur_row'], info['cur_column']]


        info['cur_row'] = info['cur_row'] + 1
        if info['cur_row'] >= info['rows']:
            info['cur_row'] = 0
            info['cur_column'] = info['cur_column'] + 1

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(axis_title)
        if x_limit is not None:
            ax.set_xlim(*x_limit)
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        ax.plot(x, y, color)


    def showPlots(self):
        if self.off: return
        plt.show()

    def reset(self, fig_title):
        if self.off: return
        fig = plt.figure(fig_title)
        plt.clf()
        self.currentPlots[fig_title] = {
            'row':1,
            'column':1
        }

    def TurnOff(self):
        self.off = True

if __name__ == "__main__":
    import numpy as np
    t = np.arange(1,30)
    y = np.arange(1,30) * 5
    PL = PlotManager(1)
    PL.addPlot("a", "t", "y", t, y)
    PL.showPlots()
    input('-')
