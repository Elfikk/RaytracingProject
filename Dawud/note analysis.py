#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 19:33:51 2021

@author: dawudabd-alghani
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.io.wavfile as wav

def load_wav(filename):
    rate, signal = wav.read(filename,mmap=False)    
    if len(signal.shape) > 1:
        signal = signal[...,0]        
    time = np.arange(len(signal))/rate
    signal = signal - np.mean(signal)
    signal=norm(signal)
    return signal, rate, time

def norm(signal):
    signal = signal[:].astype(np.float32)
    norm = np.abs(signal).max()
    signal /=norm
    return signal

data, rate, time = load_wav('note a.wav')
fft_data = np.fft.rfft(data) 
freqs = np.fft.rfftfreq(len(data),d=1./rate)

#Find the peak frequency in the data
freq_peak = freqs[np.argmax(np.abs(fft_data))]

#find the nearest note using the equal tempered 
#scale with middle A=440 Hz. This is half steps
note_step = np.log(freq_peak/261.63)/(np.log(2.)/12.)
notes = ['C','C#/Db', 'D','D#/Eb','E','F','F#/Gb','G','G#/Ab','A','A#/Bb', 'B']
print("Peak Frequency: {:.2f} Hz".format(freq_peak))

#Work out octave and note
octave = 4 + np.int(note_step/12.)
note = np.int(np.rint(note_step-12*np.int(note_step/12.)))
if note == 12: note = 0
print('Peak Note: {} octave: {}'.format(notes[note],octave))

nearest_freq = 261.63*(2.)**(np.rint(note_step)/12)
harms = nearest_freq * (2.)**(np.arange(-24,24)/12.)

#Audio time stream
plt.plot(time,data,label='Audio Sample')
plt.xlabel(r'time [s]')
plt.ylabel(r'Amplitude')
plt.legend()
plt.show()

plt.plot(freqs,np.real(fft_data),label='Real FFT coeffs')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

plt.plot(freqs,np.abs(fft_data)**2/len(data),label='Power Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('$Power$')
plt.xscale('log')
plt.legend()
plt.minorticks_on()
plt.grid(which='both',alpha=0.5)
plt.show()

plt.plot(freqs,np.abs(fft_data)**2/len(data),label='Power Spectrum')
plt.yscale('log')
plt.xlabel('Frequency [Hz]')
plt.ylabel('$Power$')
plt.xscale('log')
plt.legend()
plt.minorticks_on()
plt.grid(which='both',alpha=0.5)
plt.show()

plt.ylabel('Power')
mask = ((freqs > harms[0]) & (freqs < harms[-1]))
plt.plot(freqs[mask],np.abs(fft_data[mask])/np.sqrt(len(data)),label='Power Spectrum')
for f in harms:
    plt.axvline(f,zorder=4,ls='--',c='C2',alpha=0.7)
plt.xlabel('Frequency [Hz]')
plt.xlim([harms[0],harms[-1]])
plt.axvline(freq_peak,zorder=4,ls='--',c='C3')
plt.show()