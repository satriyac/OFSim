from collections import ChainMap
from django.shortcuts import render, HttpResponse
import Simulation.module as m
import Simulation.plot as p
from .models import Information
from django.http import JsonResponse
import random
import string
from .forms import ParameterForm
from django.conf import settings
import base64

# Create your views here.
txt = {
    'ofdmt': Information.objects.filter(Title='OFDM Transmitter - Materi'),
    'ofdmc': Information.objects.filter(Title='OFDM Channel - Materi'),
    'ofdmr': Information.objects.filter(Title='OFDM Receiver - Materi'), }


def ofdmsim(request):
    txt_home = {'active_tab': 'nav-ofdmt-tab'}
    return render(request, 'ofdmsim/ofdmsim.html', dict(ChainMap(txt_home, txt)))


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def ofdmTransceiver(request):
    if request.method == 'POST':
        form = ParameterForm(request.POST)
        if form.is_valid():
            input_text = form.cleaned_data['input_text']
            # Get values of subcarrier and mod_order from input text
            n_subpembawa = 0
            n_pilot = 0
            nilai_pilot = 0+0j
            panjang_cp = 0
            snr_db = 0
            bit_tiap_simbol = 0

            # assign module variable with 0
            pembawa = None
            subpembawa_pilot = None
            subpembawa_data = None
            panjang_bit = None
            bit_paralel = None
            bit_serial = None
            peta_konstelasi = None
            tabel_pengawapeta = None
            tabel_pemeta = None
            simbol_ranah_frekuensi = None
            simbol_ranah_waktu = None
            simbol_terkirim = None
            respon_kanal = None
            simbol_terima = None
            simbol_terima_tanpacp = None
            simbol_terima_ranah_frekuensi = None
            estimasi_pada_pilot = None
            kanal_terestimasi = None
            simbol_terekualisasi = None
            konstelasi_terima = None
            bit_paralel_terima = None
            penentu_keras = None
            bit_serial_terima = None

            # assign imagepath with 0
            subcarriers_img_base64 = None
            bit_generator_img_base64 = None
            constellation_map_img_base64 = None
            ifft_output_img_base64 = None
            channel_response_img_base64 = None
            ofdm_txrx_img_base64 = None
            channel_estimation_img_base64 = None
            received_const_img_base64 = None
            equalizer_const_img_base64 = None
            received_bits_img_base64 = None

            for param in input_text.split():
                try:
                    error_message = ""

                    if param.startswith('ns='):
                        n_subpembawa = int(param.split('=')[1])
                    elif param.startswith('np='):
                        n_pilot = int(param.split('=')[1])
                    elif param.startswith('pv='):
                        nilai_pilot = complex(param.split('=')[1])
                    elif param.startswith('bps='):
                        bit_tiap_simbol = int(param.split('=')[1])
                    elif param.startswith('cp='):
                        panjang_cp = int(param.split('=')[1])
                    elif param.startswith('snr='):
                        snr_db = int(param.split('=')[1])

                    # Fungsi menampilkan Plot
                    elif param.startswith('bit-generate'):
                        pembawa, subpembawa_pilot, subpembawa_data, panjang_bit = m.konfigurasi_subpembawa(
                            n_subpembawa, n_pilot, bit_tiap_simbol)
                        bit_serial = m.pembangkit_bit(panjang_bit)
                        subcarriers_img_base64 = p.get_plot_subcarriers(
                            subpembawa_pilot, subpembawa_data, n_subpembawa)
                        bit_generator_img_base64 = p.get_plot_bitgenerate(
                            bit_serial, bit_tiap_simbol)

                    elif param.startswith('tx-serial-parallel'):
                        bit_paralel = m.serial_ke_paralel(
                            bit_serial, subpembawa_data, bit_tiap_simbol)

                    elif param.startswith('digital-modulation'):
                        peta_konstelasi, tabel_pengawapeta, tabel_pemeta = m.pemeta(
                            bit_paralel, bit_tiap_simbol)
                        constellation_map_img_base64 = p.get_plot_constellation_map(
                            peta_konstelasi, tabel_pemeta, bit_tiap_simbol)

                    elif param.startswith('ifft'):
                        simbol_ranah_frekuensi = m.simbol_ofdm(
                            n_subpembawa, peta_konstelasi, subpembawa_pilot, nilai_pilot, subpembawa_data)
                        simbol_ranah_waktu = m.idft(simbol_ranah_frekuensi)
                        ifft_output_img_base64 = p.get_plot_ifft_output(
                            simbol_ranah_waktu)

                    elif param.startswith('add-cp'):
                        simbol_terkirim = m.penambahan_cp(
                            simbol_ranah_waktu, panjang_cp)

                    elif param.startswith('ofdm-channel'):
                        respon_kanal, simbol_terima = m.kanal(
                            n_subpembawa, simbol_terkirim, snr_db)
                        channel_response_img_base64 = p.get_plot_channel_response(
                            pembawa, respon_kanal, n_subpembawa)
                        ofdm_txrx_img_base64 = p.get_plot_ofdm_txrx(
                            simbol_terkirim, simbol_terima)

                    # elif param.startswith('awgn'):
                    #    ofdm_txrx_img_base64 = p.get_plot_ofdm_txrx(
                    #        ofdm_tx, ofdm_rx)

                    elif param.startswith('remove-cp'):
                        simbol_terima_tanpacp = m.penghapusan_cp(
                            simbol_terima, panjang_cp, n_subpembawa)

                    elif param.startswith('fft'):
                        simbol_terima_ranah_frekuensi = m.dft(
                            simbol_terima_tanpacp)

                    elif param.startswith('equalizer'):
                        estimasi_pada_pilot, kanal_terestimasi = m.estimasi_kanal(
                            simbol_terima_ranah_frekuensi, subpembawa_pilot, nilai_pilot, pembawa)
                        channel_estimation_img_base64 = p.get_plot_channel_estimation(
                            pembawa, respon_kanal, subpembawa_pilot, estimasi_pada_pilot, kanal_terestimasi)
                        simbol_terekualisasi = m.ekualiser(
                            simbol_terima_ranah_frekuensi, kanal_terestimasi)
                        konstelasi_terima = m.hapus_pilot(
                            simbol_terekualisasi, subpembawa_data)
                        received_const_img_base64 = p.get_plot_received_const(
                            konstelasi_terima)
                        bit_paralel_terima, penentu_keras = m.pengawapeta(
                            konstelasi_terima, tabel_pengawapeta)
                        equalizer_const_img_base64 = p.get_plot_equalizer_const(
                            konstelasi_terima, penentu_keras, tabel_pengawapeta, peta_konstelasi)

                    elif param.startswith('rx-parallel-serial'):
                        bit_serial_terima = m.paralel_ke_serial(
                            bit_paralel_terima)

                    elif param.startswith('received-bits'):
                        received_bits_img_base64 = p.get_plot_received_bits(
                            bit_serial_terima, bit_serial, bit_tiap_simbol)

                except Exception as e:
                    print(e)
                    error_message = "Kode yang anda masukkan salah"

            print(n_subpembawa, n_pilot, nilai_pilot,
                  bit_tiap_simbol, panjang_cp, snr_db)
            # OFDM Function

            # Send HTML response with plot images as data URIs
            response_data = {
                'subcarriers_img': subcarriers_img_base64,
                'bit_generator_img': bit_generator_img_base64,
                'constellation_map_img': constellation_map_img_base64,
                'ifft_output_img': ifft_output_img_base64,
                'channel_response_img': channel_response_img_base64,
                'ofdm_txrx_img': ofdm_txrx_img_base64,
                'channel_estimation_img': channel_estimation_img_base64,
                'received_const_img': received_const_img_base64,
                'equalizer_const_img': equalizer_const_img_base64,
                'received_bits_img': received_bits_img_base64,
                'error_message_text': error_message
            }

            # Return the response as JSON
            return JsonResponse(response_data)
    else:
        form = ParameterForm()

    return render(request, 'ofdmsim/ofdmsim.html', {'form': form})
