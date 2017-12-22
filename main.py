from appJar import gui
import numpy as np
from plotManager import PlotManager
from cep import real_cepstrum, complex_cepstrum
from peakdetect import peakdet
PL = PlotManager(6)

#Important reference for cepstrum and signal genearation:
#https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py

def openFile(button):
    print("File opened.")

def generateWave(button):
    fundamental = app.getEntry("fundamental") or 100
    numberOfSins = int(app.getEntry("numberOfSins")) or 400
    duration = app.getEntry("duration") or 5
    fs = int(app.getEntry("sampleRate")) or 8000
    samples = int(fs*duration)
    t = np.arange(samples) / fs

    print(fundamental)
    print(numberOfSins)
    harmonics = np.arange(1,numberOfSins) * fundamental
    print(len(harmonics))
    signal = np.sin(2.0*np.pi*harmonics[:,None]*t).sum(axis=0)
    signal = 10*signal
    #app.addPlot("time", t, signal)
    fig_title = "spectral envelope"
    PL.reset(fig_title)

    PL.addPlot("spectral envelope", "time", "time (seconds)", "amplitude", t, signal)

    fftDict = getFFT(signal, fs)
    PL.addPlot(
        fig_title,
        "frequency",
        "frequency (hertz)",
        "power",
        fftDict['scale'],
        fftDict['fft'])

    N = len(signal)
    cepsDict = getCepstrum(fftDict['full'], N)
    for key in cepsDict:
        print(key, len(cepsDict[key]))

    PL.addPlot(
        fig_title,
        "log no phase frequency",
        "frequency (hertz)",
        "log power",
        fftDict['scale'],
        cepsDict['halfLog']
    )

    cep_tup = complex_cepstrum(signal)

    PL.addPlot(
        fig_title,
        "log phased frequency",
        "frequency (hertz)",
        "log power",
        fftDict['scale'],
        2/N * cep_tup[2][:N//2]
    )

    PL.addPlot(
        fig_title,
        "complex cepstrum",
        "time (quefrency seconds)",
        "amplitude",
        t,
        cep_tup[0],
        x_limit=(0.0, 0.5),
        y_limit=(-5., +10.)
    )

    PL.addPlot(
        fig_title,
        "power cepstrum",
        "time (quefrency seconds)",
        "amplitude",
        t,
        cepsDict['cepstrum'],
        x_limit=(0.0, 0.5),
        y_limit=(0.0, 0.2)
    )

    PL.showPlots()

    print("Wave being generated.")
    generateSourceFilter(cep_tup[0], t, "source - filter")
    generateSourceFilter(cepsDict['cepstrum'], t, "source - filter power")

def generateSourceFilter(cepstrum, scale, figTitle):
    fs = int(app.getEntry("sampleRate")) or 8000
    cutOff = app.getEntry('sourceCutOff') or 170.0

    timePeriod = 1.0/cutOff
    cutOffIndex = len(scale) - 1
    for index, value in enumerate(scale):
        if value >= timePeriod:
            cutOffIndex = index
            break

    source = cepstrum[:cutOffIndex]
    source_scale = scale[:cutOffIndex]

    _filter = cepstrum[cutOffIndex:]
    _filter_scale = scale[cutOffIndex:]

    #PL.reset(figTitle)
    PL.startPlot(figTitle, 2,2)
    PL.addPlotN(
        figTitle,
        "source",
        "time (quefrency)",
        "amplitude",
        source_scale,
        source
    )

    PL.addPlotN(
        figTitle,
        "filter",
        "time (quefrency)",
        "amplitude",
        _filter_scale,
        _filter
    )


    source = np.abs(source)
    _filter = np.abs(_filter)

    source_fft_dict = getFFT(source, fs)
    filter_fft_dict = getFFT(_filter, fs)

    PL.addPlotN(
        figTitle,
        "source f domain",
        "cepstral frequency",
        "amplitude",
        source_fft_dict['scale'],
        source_fft_dict['fft']
    )

    PL.addPlotN(
        figTitle,
        "filter f",
        "cepstral frequency",
        "amplitude",
        filter_fft_dict['scale'],
        filter_fft_dict['fft']
    )
    PL.showPlots()

    createDiscrete('filter', _filter, _filter_scale)

def createDiscrete(figTitle, cepstrum, scale):

    delta = 0.01

    cepstrum = np.abs(cepstrum)
    peak_tuples = peakdet(cepstrum, delta)[0]
    indices = [x[0] for x in peak_tuples]
    new_scale = [1.0/x for x in scale]
    y = np.zeros(len(cepstrum))
    print(peak_tuples)
    for peak in peak_tuples:
        y[int(peak[0])] = peak[1]

    for i, val in enumerate(scale):
        print(val, new_scale[i])
        if i > 10:
            break
    PL.reset(figTitle)
    PL.addPlot(
        figTitle,
        "Discrete Points",
        "frequency",
        "amplitude",
        new_scale,
        y,
        "b+"
    )

#A reference: https://stackoverflow.com/a/30077972
def getFFT(signal, samplingFreq):
    N = len(signal)
    print(N)
    fft = np.fft.fft(signal)
    xf = np.linspace(0.0, samplingFreq/2, len(signal)/2)
    print(len(fft[:N//2]))
    return {
        'full':fft,
        'fft': 2/N * np.abs(fft[:N//2]),
        'scale': xf
    }

#assumes fft is absolute.
def getCepstrum(fft, N):
    absFft = np.abs(fft.real)
    log = 2.0 * np.log(absFft)
    #log = np.log(absFft)
    ifft = np.fft.ifft(log)
    print(ifft)
    return {
        'log':log,
        'halfLog': 2/N * log[:N//2],
        'cepstrum':np.square(np.abs(ifft))
        #'cepstrum':ifft.real
    }



app = gui("Frequency Estimation and Envelope Estimation")
app.setSticky("news")
app.setExpand("both")
app.setFont(20)
app.addLabel("title", "Spectral Envelope Estimator")

app.addLabel("GenerationL", "Either open a file or generate a wave", row=0, column=0, colspan=2)
app.addButton("Open File", func=openFile, row=1, column=0)
app.addButton("Generate Wave", func=generateWave, row=1, column=1)

app.addHorizontalSeparator(2,0,3, colour="blue")
app.addLabel("subHeading1", "Generation Parameters", row=3, colspan=3)
app.addLabelNumericEntry("fundamental", row=4, column=0)
app.addLabelNumericEntry("numberOfSins", row=4, column=1)
app.addLabelNumericEntry("duration", row=4, column=2)

app.addHorizontalSeparator(5,0,2, colour="blue")
app.addLabel("subHeading2", "Analysis Parameters", row=5, colspan=2)
app.addLabelNumericEntry("sampleRate", row=6, column=1)

app.addHorizontalSeparator(6,0,2, colour="blue")
app.addLabel("subHeading3", "Cepstrum Analysis Parameters", row=6, colspan=2)
app.addLabelNumericEntry("sourceCutOff", row=6, column=1)

#app.addButtons(["Open File",  "Generate Wave"], "row=0" )



#
# app.setLabelBg("title", "red")

app.go()
