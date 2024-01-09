import numpy as np
import scipy.interpolate

""" Modul Configurasi Subcarrier """
# Fungsi konfigurasi subcarrier


def konfigurasi_subpembawa(n_subpembawa, n_pilot, bit_tiap_simbol):
    pembawa = np.arange(n_subpembawa)
    subpembawa_pilot = pembawa[::n_subpembawa//n_pilot]
    subpembawa_pilot = np.hstack([subpembawa_pilot, np.array([pembawa[-1]])])
    subpembawa_data = np.delete(pembawa, subpembawa_pilot)
    panjang_bit = len(subpembawa_data)*bit_tiap_simbol
    return pembawa, subpembawa_pilot, subpembawa_data, panjang_bit


""" Modul Transmitter """
# Blok pembangkitan data bit sejumlah panjangBit


def pembangkit_bit(panjangBit):
    bit_serial = np.random.binomial(n=1, p=0.5, size=(panjangBit, ))
    return bit_serial

# Blok serial ke paralel


def serial_ke_paralel(bit_serial, subpembawa_data, bit_tiap_simbol):
    bit_paralel = bit_serial.reshape((len(subpembawa_data), bit_tiap_simbol))
    return bit_paralel

# Blok pemeta konstelasi


def pemeta(bits, bit_tiap_simbol):

    pemeta_bpsk = {
        (0,): -4.24264+0.0j,
        (1,):  4.24264+0.0j,
    }
    pemeta_qpsk = {
        (0, 0): -3-3j,
        (1, 0):  3-3j,
        (0, 1): -3+3j,
        (1, 1):  3+3j
    }
    pemeta_8qam = {
        (0, 0, 0): (-3+1j)*(4.2426/3.1623),
        (1, 0, 0): (-3-1j)*(4.2426/3.1623),
        (0, 1, 0): (-1+1j)*(4.2426/3.1623),
        (1, 1, 0): (-1-1j)*(4.2426/3.1623),
        (0, 0, 1): (3+1j)*(4.2426/3.1623),
        (1, 0, 1): (3-1j)*(4.2426/3.1623),
        (0, 1, 1): (1+1j)*(4.2426/3.1623),
        (1, 1, 1): (1-1j)*(4.2426/3.1623)
    }
    pemeta_16qam = {
        (0, 0, 0, 0): -3-3j,
        (0, 0, 0, 1): -3-1j,
        (0, 0, 1, 0): -3+3j,
        (0, 0, 1, 1): -3+1j,
        (0, 1, 0, 0): -1-3j,
        (0, 1, 0, 1): -1-1j,
        (0, 1, 1, 0): -1+3j,
        (0, 1, 1, 1): -1+1j,
        (1, 0, 0, 0):  3-3j,
        (1, 0, 0, 1):  3-1j,
        (1, 0, 1, 0):  3+3j,
        (1, 0, 1, 1):  3+1j,
        (1, 1, 0, 0):  1-3j,
        (1, 1, 0, 1):  1-1j,
        (1, 1, 1, 0):  1+3j,
        (1, 1, 1, 1):  1+1j
    }
    if bit_tiap_simbol == 1:
        tabel_pemeta = pemeta_bpsk
    elif bit_tiap_simbol == 2:
        tabel_pemeta = pemeta_qpsk
    elif bit_tiap_simbol == 3:
        tabel_pemeta = pemeta_8qam
    elif bit_tiap_simbol == 4:
        tabel_pemeta = pemeta_16qam
    peta_konstelasi = np.array([tabel_pemeta[tuple(b)] for b in bits])
    tabel_pengawapeta = {v: k for k, v in tabel_pemeta.items()}
    return peta_konstelasi, tabel_pengawapeta, tabel_pemeta

# Pembentukan simbol OFDM dengan mengalokasikan data pada subpembawa data, dan pilot pada subpembawa pilot


def simbol_ofdm(n_subpembawa, peta_konstelasi, subpembawa_pilot, nilai_pilot, subpembawa_data):
    simbol_dom_frek = np.zeros(n_subpembawa, dtype=complex)
    simbol_dom_frek[subpembawa_pilot] = nilai_pilot
    simbol_dom_frek[subpembawa_data] = peta_konstelasi
    return simbol_dom_frek

# Blok IDFT untuk mengubah ke ranah waktu


def idft(simbol_dom_frek):
    simbol_dom_waktu = np.fft.ifft(simbol_dom_frek)
    return simbol_dom_waktu

# Blok penambahan cyclic prefix


def penambahan_cp(simbol_dom_waktu, panjang_cp):
    # take the last CP samples ...
    cyclic_prefix = simbol_dom_waktu[-panjang_cp:]
    # ... and add them to the beginning
    simbol_terkirim = np.hstack([cyclic_prefix, simbol_dom_waktu])
    return simbol_terkirim


""" Modul Channel """
# Kanal multipoth model two-tap delayed line


def kanal(n_subpembawa, simbol_terkirim, snr_db):
    respon_impuls = np.array([1, 0, 0.3+0.3j])
    respon_kanal = np.fft.fft(respon_impuls, n_subpembawa)
    terkonvolusi = np.convolve(simbol_terkirim, respon_impuls)
    daya_isyarat = np.mean(abs(terkonvolusi**2))
    daya_derau = daya_isyarat * 10**(-snr_db/10)
    derau_putih = np.sqrt(daya_derau/2) * (np.random.randn(*terkonvolusi.shape) +
                                           1j*np.random.randn(*terkonvolusi.shape))
    return respon_kanal, terkonvolusi + derau_putih


""" Modul Receiver """
# Blok penghapusan cyclic prefix


def penghapusan_cp(isyarat_terima, cyclic_prefix, pembawa):
    return isyarat_terima[cyclic_prefix:(cyclic_prefix+pembawa)]

# Transform to frequency domain


def dft(simbol_terima):
    simbol_dom_frek = np.fft.fft(simbol_terima)
    return simbol_dom_frek

# Blok estimasi kanal


def estimasi_kanal(simbol_dom_frek, subpembawa_pilot, nilai_pilot, pembawa):
    pilot_terima = simbol_dom_frek[subpembawa_pilot]
    estimasi_pada_pilot = pilot_terima / nilai_pilot
    estimasi_mutlak = scipy.interpolate.interp1d(subpembawa_pilot, abs(
        estimasi_pada_pilot), kind='linear', fill_value="extrapolate")(pembawa)
    estimasi_fase = scipy.interpolate.interp1d(subpembawa_pilot, np.angle(
        estimasi_pada_pilot), kind='linear', fill_value="extrapolate")(pembawa)
    kanal_terestimasi = estimasi_mutlak * np.exp(1j*estimasi_fase)
    return estimasi_pada_pilot, kanal_terestimasi

# Blok Ekualiser


def ekualiser(simbol_dom_frek, kanal_terestimasi):
    simbol_terekualisasi = simbol_dom_frek / kanal_terestimasi
    return simbol_terekualisasi

# Blok Penghapusan pilot


def hapus_pilot(simbol_terekualisasi, subpembawa_data):
    konstelasi_terima = simbol_terekualisasi[subpembawa_data]
    return konstelasi_terima

# Blok Pengawapeta


def pengawapeta(konstelasi_terima, tabel_pengawapeta):
    konstelasi = np.array([x for x in tabel_pengawapeta.keys()])
    dists = abs(konstelasi_terima.reshape((-1, 1)) -
                konstelasi.reshape((1, -1)))
    indeks_konstelasi = dists.argmin(axis=1)
    penentu_keras = konstelasi[indeks_konstelasi]
    bit_paralel_terima = np.vstack(
        [tabel_pengawapeta[T] for T in penentu_keras])
    return bit_paralel_terima, penentu_keras

# Blok paralel ke serial


def paralel_ke_serial(bit_paralel_terima):
    bit_serial_terima = bit_paralel_terima.reshape((-1,))
    return bit_serial_terima
