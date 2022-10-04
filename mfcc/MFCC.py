"""
Mel-Frequency Cepstral Coefficients algorithm implementation
Created for lectures of KKY/ARŘ at University of West Bohemia
"""

import numpy as np
from scipy.io import wavfile
import scipy.fftpack as fft
import scipy.signal
import csv
from matplotlib import pyplot as plt

"""
Audio normalization implemented to mimic Matlab normalization
"""
def normalize_audio(original_audio):
    #normalized_audio = original_audio / np.max(np.abs(original_audio))
    normalized_audio = original_audio / 2**15
    return normalized_audio

"""
Convert frequency in Hertz to frequency in Mel
"""
def hz2mel(frequency):
    mel_frequency = 2595 * np.log10(1+frequency/700)
    return mel_frequency


"""
Convert frequency in Mel to frequency in Hertz
"""
def mel2hz(mel_frequency):
    frequency = 700*((10**(mel_frequency/2595))-1)
    return frequency


"""
Table of all relevant points of all the triangular filters
Each filter starts at point (filter_id -1), peaks at point (filter_id) and ends at point (filter_id+1)
"""
def filter_table(max_frequency_Hz = 4000, number_of_filters = 15):
    step = hz2mel(max_frequency_Hz)/(number_of_filters+1)
    filter_centers_mel = np.zeros(number_of_filters+2)
    filter_centers_Hz = np.zeros(number_of_filters+2)
    center = step
    for i in range(1,number_of_filters+2):
        filter_centers_mel[i] = center
        filter_centers_Hz[i] = mel2hz(center)
        center = center + step
        
    """
    for i,h,m in zip(range(1,17),filter_centers_Hz,filter_centers_mel):
        print("{i} \t {m:.4f} \t {h:.4f}".format(i=i,h=h,m=m))
    """
    return filter_centers_Hz


"""
Formula to calculate the value of the chosen filter at the chosen frequency
"""
def filter_function(frequency,filter_start,filter_center,filter_end):
    if ((filter_start<=frequency) and (frequency<filter_center)):
        return ((frequency-filter_start)/(filter_center-filter_start))
    elif ((filter_center<=frequency) and (frequency<filter_end)):
        return ((frequency-filter_end)/(filter_center-filter_end))
    else: 
        return 0


"""
Value of chosen triangular filter at a given frequency calculated with a corresponding formula
"""
def filter_value(frequency,filter_table,filter_id=1):
    filter_start = filter_table[filter_id-1]
    filter_center = filter_table[filter_id]
    filter_end = filter_table[filter_id+1]
    
    filter_value = filter_function(frequency,filter_start,filter_center,filter_end)
    return filter_value

#print(filter_table())

#"""
big_table = np.zeros((15,128))
step = 4000/128
table = filter_table()
for i in range(15):
    f = 0
    val = filter_value(f,table,i+1)
    for j in range(128):
        big_table[i,j] = val
        f += step
        val = filter_value(f,table,i+1)
        
#"""

print(np.shape(big_table))
with open("./out.csv", 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    for row in big_table:
        csvwriter.writerow(row)

# 0.032 s * 8000 Hz = 256
# 0.010 s * 8000 Hz = 80

"""
Split source audio into segments of size 'segment_size' with a shift between segments the size of 'shift'
"""
def microsegments(sampling_frequency, audio_signal, segment_size = 256, shift = 10):
    shift_size = shift / 1000 * sampling_frequency
    number_of_segments = int((len(audio_signal) - segment_size) / shift_size) + 1
    
    segments = []
    start = 0
    end = start + segment_size
    for i in range(number_of_segments):   
        segments.append(audio_signal[start:end])
        start = start + int(shift_size) 
        end = start + segment_size
    return segments

def mfcc(audio_path, segment_size = 256, shift = 10):
    sampling_frequency, audio = wavfile.read(audio_path)
    normalized_audio = normalize_audio(audio)
    segments = microsegments(sampling_frequency, normalized_audio)
    
    mfcc_table = []
    window = scipy.signal.get_window("hamm", segment_size, fftbins=True)
    for segment in segments:
        windowed_segment = segment * window 
        segment_fft = fft.fft(windowed_segment)
        absolute_value = np.abs(segment_fft)
        first_half = first_abs[0:(segment_size/2)] #we only need the first half as the fft is symmetric by y-axis
        result = np.dot(big_table,first_half)
        log_result = np.log10(result)
        cos_result = fft.dct(log_result,norm='ortho') #ortho is used to mimic matlab dct functionality

        mfcc_table.append(cos_result)
    return mfcc_table


if __name__ == "__main__":
    
    AUDIO_PATH = "mfcc/00010001.wav"

    table = mfcc(AUDIO_PATH)
    print(table[0])
    #sampling_frequency, audio = wavfile.read(AUDIO_PATH)
    exit()
    sampling_frequency, audio = wavfile.read(SOUND_PATH)
    normalized_audio = normalize_audio(audio)
    segments = microsegments(8000, normalized_audio)

    first_segment = segments[0]
    window = scipy.signal.get_window("hamm", 256, fftbins=True)
    first_hamm = first_segment * window 
    first_fft = fft.fft(first_hamm)
    first_abs = np.abs(first_fft)
    first_half = first_abs[0:128]

    result = np.dot(big_table,first_half)
    log_result = np.log10(result)
    cos_result = fft.dct(log_result,norm='ortho')
    print(cos_result)
    #result = big_table @ first_half
    #plt.plot(first_half)
    #plt.show()
    """
    #print("Sample rate: {0}Hz".format(sampling_frequency))
    #print("Audio duration: {0}s".format(len(audio) / sampling_frequency))

    with open("./out2.csv", 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        for row in segments:
            csvwriter.writerow(row)






"""