import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import scipy.signal as sg
import scipy.fftpack as fr
import cmath as c
import Simulation.plot as p


""" Error Handling dari Input User 
# Fungsi untuk cek data parameter

def inputData(data):
    # Dictionary dari kata untuk input kode

    dictOFDM = ['bits-per-symbol', 'cycle-prefix', 'n-pilot', 'n-subcarriers', 'ofdm-transceiver', 'pilot-value', 'snr-db']
    # Error handling modul Signal Generator
    # signal-generator amplitude 1 frequency 1000 freq-sampling 100
    # show signal
   
    
    if data[0] == 'ofdm-transceiver':
        n_subcarriers       = int(data[2])
        n_pilot             = int(data[4])
        pilot_value         = complex(data[6])
        bits_per_symbol     = int(data[8])
        cycle_prefix        = int(data[10])
        snr_db              = int(data[12])

        show                = ofdm_transceiver(n_subcarriers, n_pilot, pilot_value,  bits_per_symbol, cycle_prefix, snr_db)
        remove              = [data[2], data[4], data[6], data[8], data[10], data[12]]
        final_list          = list(set(data) - set(remove))
        final_list.sort()
        
        cek_kata            = (final_list == dictOFDM)
        cek = print(final_list)
        dic = print(dictOFDM)      
   
    return cek_kata, show
"""


""" Modul Configurasi Subcarrier """
# Fungsi konfigurasi subcarrier
def subcarrier_config(n_subcarriers, n_pilot, bits_per_symbol):
    allCarriers = np.arange(n_subcarriers)
    pilotCarriers = allCarriers[::n_subcarriers//n_pilot]
    pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
    n_pilot = n_pilot + 1
    dataCarriers = np.delete(allCarriers, pilotCarriers)
    payloadBits_per_OFDM = len(dataCarriers)*bits_per_symbol
    return allCarriers, pilotCarriers, dataCarriers, payloadBits_per_OFDM


""" Modul Transmitter """
# Fungsi pembangkitan bit sejumlah payload
def bit_generator(payloadBits_per_OFDM):
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    return bits

def serial_to_paralel(bits, dataCarriers, bits_per_symbol ):
    return bits.reshape((len(dataCarriers), bits_per_symbol))

# Frequency Domain
def Mapping(bits, bits_per_symbol ):
    
    mapping_bpsk = {
    (0,) : -4.24264+0.0j,
    (1,) :  4.24264+0.0j,
}
    mapping_qpsk = {
    (0,0) : -3+3j,
    (1,0) : -3-3j,
    (0,1) :  3+3j,
    (1,1) :  3-3j
}
    mapping_8qam = {
    (0,0,0) : (-3+1j)*(4.2426/3.1623),
    (1,0,0) : (-3-1j)*(4.2426/3.1623),
    (0,1,0) : (-1+1j)*(4.2426/3.1623),
    (1,1,0) : (-1-1j)*(4.2426/3.1623),
    (0,0,1) : ( 3+1j)*(4.2426/3.1623),
    (1,0,1) : ( 3-1j)*(4.2426/3.1623),
    (0,1,1) : ( 1+1j)*(4.2426/3.1623),
    (1,1,1) : ( 1-1j)*(4.2426/3.1623)
}
    mapping_16qam = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}
    if bits_per_symbol == 1:
        mapping_table = mapping_bpsk
    elif bits_per_symbol == 2:
        mapping_table = mapping_qpsk
    elif bits_per_symbol == 3:
        mapping_table = mapping_8qam
    elif bits_per_symbol == 4:
        mapping_table = mapping_16qam
    constellation_mapp = np.array([mapping_table[tuple(b)] for b in bits])
    constellation_demapp = {v : k for k, v in mapping_table.items()}
    return constellation_mapp, constellation_demapp, mapping_table

# OFDM Symbol generate
def OFDM_symbol(n_subcarriers, QAM_payload, pilotCarriers, pilotValue, dataCarriers):
    symbol = np.zeros(n_subcarriers, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol

#IDFT untuk mengubah ke ranah waktu
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

#Penambahan cycle prefix
def addCP(OFDM_time, CP):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

""" Modul Channel """
#OFDM Channel
def channel(n_subcarriers, signal, snr_db):
    channelResponse = np.array([1, 0, 0.3+0.3j])
    H_exact = np.fft.fft(channelResponse, n_subcarriers)
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-snr_db/10)  # calculate noise power based on signal power and SNR
    
    #print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return H_exact, convolved + noise

""" Modul Receiver """
#Remove CP
def removeCP(signal,cycle_prefix, n_subcarriers):
    return signal[cycle_prefix:(cycle_prefix+n_subcarriers)]

#Transform to frequency domain
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized, dataCarriers):
    return equalized[dataCarriers]
#Channel Estimation
def channelEstimate(OFDM_demod, pilotCarriers, pilotValue, allCarriers):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear', fill_value="extrapolate" )(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear', fill_value="extrapolate")(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    return Hest_at_pilots, Hest
def Demapping(received_const, const_demapp):
    # array of possible constellation points
    constellation = np.array([x for x in const_demapp.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(received_const.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([const_demapp[C] for C in hardDecision]), hardDecision

def PS(bits):
    return bits.reshape((-1,))

"""
#Input Modulasi OFDMT
def ofdm_transceiver(n_subcarriers, n_pilot, pilot_value, bits_per_symbol, cycle_prefix, snr_db):
    allCarriers,pilotCarriers, dataCarriers, payloadBits_per_OFDM = subcarrier_config(n_subcarriers, n_pilot, bits_per_symbol)
    bits = bit_generator(payloadBits_per_OFDM)
    bits_paralel = serial_to_paralel(bits, dataCarriers, bits_per_symbol)
    const_mapp, const_demapp  = Mapping(bits_paralel, bits_per_symbol)
    ofdm_data = OFDM_symbol(n_subcarriers, const_mapp, pilotCarriers, pilot_value, dataCarriers)
    ofdm_time = IDFT(ofdm_data)
    ofdm_tx = addCP(ofdm_time, cycle_prefix)
    H_exact, ofdm_rx = channel(n_subcarriers, ofdm_tx, snr_db)
    ofdm_rx_cpremove = removeCP(ofdm_rx, cycle_prefix, n_subcarriers)
    ofdm_freq = DFT(ofdm_rx_cpremove)
    hest_at_pilots, hest = channelEstimate(ofdm_freq, pilotCarriers, pilot_value, allCarriers)
    equalized_hest = equalize(ofdm_freq, hest)
    receiced_const = get_payload(equalized_hest, dataCarriers)
    bits_received, hard_decision = Demapping(receiced_const, const_demapp)
    bits_serial = PS(bits_received)
    show = p.get_plot_ofdm(bits, pilotCarriers, dataCarriers, allCarriers, n_subcarriers, const_mapp, H_exact, ofdm_time, 
                           ofdm_tx, ofdm_rx, hest_at_pilots, hest, receiced_const, bits_received, hard_decision, bits_serial )
    #plt.step(np.arange(0, len(bits)), bits)
    return show

"""