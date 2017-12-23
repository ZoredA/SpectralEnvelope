import numpy as np
import sounddevice as sd
from appJar import gui
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
    number_of_sins = int(app.getEntry("number of sins")) or 40
    duration = app.getEntry("duration") or 5
    fs = int(app.getEntry("Sampling Frequency")) or 8000
    samples = int(fs*duration)
    t = np.arange(samples) / fs

    print(fundamental)
    harmonics = np.arange(1,number_of_sins) * fundamental
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
    #PL.TurnOff()

    print("Wave being generated.")
    # (frequencies, values) = findPeaks(cepsDict['cepstrum'], t, "Power peaks")
    # # generateSourceFilter(cep_tup[0], t, "source - filter")
    # # generateSourceFilter(cepsDict['cepstrum'], t, "source - filter power")
    # #findPeaks(cep_tup[0], t, "Complex Peaks")
    # PL.startPlot('discrete peaks', 1,1)
    # PL.addPlotN(
    #     'discrete peaks',
    #     "Discrete Points",
    #     "frequency",
    #     "amplitude",
    #     frequencies,
    #     values,
    #     "b+"
    # )
    print("Complex")
    findFundamentalViaEarliestFilter(cep_tup[0], t)
    print("Power")
    findFundamentalViaEarliestFilter(cepsDict['cepstrum'], t)

def findFundamentalViaEarliestFilter(signal, quefrency_scale):
    #First we separate the signal into source and filter. The filter
    #is the bit at the beginning, the source after.
    separated_signal = getSourceFilter(signal, quefrency_scale)
    print(separated_signal['source_scale'][0:20])
    #We look at the source part of the signal, i.e. the lower frequencies
    (frequencies, values) = findPeaks(separated_signal['source'], separated_signal['source_scale'])

    print(frequencies[0], values[1])

    return (frequencies, values)

def getSourceFilter(signal, quefrency_scale):
    fs = int(app.getEntry("Sampling Frequency")) or 8000

    #In seconds, the earliest point on the scale
    lowestQuefrency = 2.0 / fs
    cutOff = app.getEntry('Source Cut Off Frequency') or 170.0

    timePeriod = 1.0/cutOff
    cutOffIndex = len(quefrency_scale) - 1
    startIndex = None

    print(lowestQuefrency)
    print(timePeriod)

    #We can't have a cut off frequency higher than the highestFreq
    #possible frequency
    if lowestQuefrency > timePeriod:
        timePeriod = lowestQuefrency

    for index, value in enumerate(quefrency_scale):
        if startIndex is None and value >= lowestQuefrency:
            startIndex = index
        if value >= timePeriod:
            cutOffIndex = index
            break

    #high frequencies are considered filter.
    _filter = signal[startIndex:cutOffIndex]
    filter_scale = quefrency_scale[startIndex:cutOffIndex]

    #The lower frequencies, the right of the scale
    source = signal[cutOffIndex:]
    source_scale = quefrency_scale[cutOffIndex:]

    return {
        'source' : source,
        'source_scale' : source_scale,
        'filter' : _filter,
        'filter_scale' : filter_scale
    }

def findPeaks(signal, scale):
    frequency_scale = [1.0/x for x in scale]
    peaks = peakdet(signal, 0.01, frequency_scale)[0]
    frequencies = [x[0] for x in peaks]
    values = [x[1] for x in peaks]
    return (frequencies, values)

def findPeaksMultiple(signal, scale, figTitle):
    fs = int(app.getEntry("Sampling Frequency")) or 8000
    highestFreq = fs / 2
    #The smallest number that we deem a valid point before which is likely Noise
    lowestQuefrency = 1.0 / highestFreq
    cutOff = app.getEntry('Source Cut Off Frequency') or 170.0

    timePeriod = 1.0/cutOff
    cutOffIndex = len(scale) - 1
    startIndex = None

    for index, value in enumerate(scale):
        if value >= timePeriod:
            cutOffIndex = index
            break
        if startIndex is None and value >= lowestQuefrency:
            startIndex = index

    new_scale = [1.0/x for x in scale[startIndex:]]
    peak_tuples = peakdet(signal[startIndex:], 0.0001, new_scale)[0]
    peak_frequencies = [x[0] for x in peak_tuples]
    m = findMultiples(peak_frequencies, True)
    z = sorted(peak_frequencies)
    print(figTitle + '---------')
    print(z[0])
    print(z[-1])
    maximum_index = 0
    max_val = -1
    for val in m:
        if val > 40.0 and val < 110.0:
            print(val, m[val])
        if len(m[val]) > max_val:
            max_val = len(m[val])
            maximum_index = val
    print(maximum_index)
    print(len(m[maximum_index]))
    print(m[maximum_index][-1])

    print(figTitle + '---------\\')

def generateSourceFilter(cepstrum, scale, figTitle):
    fs = int(app.getEntry("Sampling Frequency")) or 8000
    highestFreq = fs / 2
    #The smallest number that we deem a valid point before which is likely Noise
    lowestQuefrency = 1.0 / highestFreq

    cutOff = app.getEntry('Source Cut Off Frequency') or 170.0

    timePeriod = 1.0/cutOff
    cutOffIndex = len(scale) - 1
    startIndex = None
    for index, value in enumerate(scale):
        if value >= timePeriod:
            cutOffIndex = index
            break
        if startIndex is None and value >= lowestQuefrency:
            startIndex = index


    source = cepstrum[startIndex:cutOffIndex]
    source_scale = scale[startIndex:cutOffIndex]

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

    createDiscrete('Discrete Points From filter', _filter, _filter_scale)

def createDiscrete(figTitle, cepstrum, scale):

    delta = 0.01

    cepstrum = np.abs(cepstrum)
    peak_tuples = peakdet(cepstrum, delta)[0]
    indices = [x[0] for x in peak_tuples]
    new_scale = [1.0/x for x in scale]
    y = np.zeros(len(cepstrum))

    peak_frequencies = []
    for peak in peak_tuples:
        y[int(peak[0])] = peak[1]
        # peak_frequencies.append( new_scale[int(peak[0])] )

    PL.startPlot(figTitle, 1,1)
    PL.addPlotN(
        figTitle,
        "Discrete Points",
        "frequency",
        "amplitude",
        new_scale,
        y,
        "b+"
    )


#Calculates the FFT of a signal. The sampling frequency
#is used to generate the list of frequencies that
#can be plotted against.
#Returns a dictionary containing
#the full fft ('full'), half the FFT or the real
#part ('fft') and the frequencies corresponding to half the
#fft, ('scale')
#A reference: https://stackoverflow.com/a/30077972
def getFFT(signal, samplingFreq):
    N = len(signal)
    print("Getting fft for signal of size ", N)
    fft = np.fft.fft(signal)
    xf = np.linspace(0.0, samplingFreq/2, len(signal)/2)
    return {
        'full':fft,
        'fft': 2/N * np.abs(fft[:N//2]),
        'scale': xf
    }

#Takes in an FFT as well as an N (signal size)
#to create the half log version.
def getCepstrum(fft, N):
    absFft = np.abs(fft)
    log = 2.0 * np.log(absFft)
    ifft = np.fft.ifft(log).real
    return {
        'log':log,
        'halfLog': 2/N * log[:N//2],
        'cepstrum':np.square(np.abs(ifft))
        #'cepstrum':ifft.real
    }

#if twoOrMore is set to True, entries that
#only have themselves as a multiple are not included in the return result
def findMultiples(numbers, twoOrMore=False):
    multiples = {}
    #Sort from low to high
    numbers = sorted(numbers)
    for num in numbers:
        if num < 31.1:
            continue
        added = False
        for base in multiples:
            if checkIfMultiple(base, num):
                multiples[base].append(num)
                added = True
        multiples[num] = [num]
    if twoOrMore is True:
        return {x:multiples[x] for x in multiples if len(multiples[x]) > 1}
    return multiples

#checks if number 2 is a multiple of number 1
def checkIfMultiple(number1, number2, error=0.001):
    rounded = np.round(number2/number1)
    actual = number2/number1
    if np.abs(actual-rounded) <= error:
        return True
    return False

def playSound(signal, fs):
    sd.play(signal, fs)

def stopSound(signal, fs):
    sd.stop()


def fillInDefaults(app):
    defaults = {
        'fundamental' : 100.0,
        'number of sins' : 40,
        'duration' : 5,
        'Sampling Frequency' : 8000,
        'Source Cut Off Frequency' : 170
    }
    for key in defaults:
        app.setEntryDefault(key, defaults[key])

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
app.addLabelNumericEntry("number of sins", row=4, column=1)
app.addLabelNumericEntry("duration", row=4, column=2)

app.addHorizontalSeparator(5,0,2, colour="blue")
app.addLabel("subHeading2", "Analysis Parameters", row=5, colspan=2)
app.addLabelNumericEntry("Sampling Frequency", row=6, column=1)

app.addHorizontalSeparator(7,0,2, colour="blue")
app.addLabel("subHeading3", "Cepstrum Analysis Parameters", row=8, colspan=2)
app.addLabelNumericEntry("Source Cut Off Frequency", row=9, column=1)

#app.addButtons(["Open File",  "Generate Wave"], "row=0" )

fillInDefaults(app)

#
# app.setLabelBg("title", "red")

app.go()
