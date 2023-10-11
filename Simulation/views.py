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
    'ofdmt' : Information.objects.filter(Title='OFDM Transmitter - Materi'),
    'ofdmc' : Information.objects.filter(Title='OFDM Channel - Materi'),
    'ofdmr' : Information.objects.filter(Title='OFDM Receiver - Materi'),}

def ofdmsim(request):
    txt_home = {'active_tab':'nav-ofdmt-tab'}
    return render(request, 'ofdmsim/ofdmsim.html', dict(ChainMap(txt_home,txt)))

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def ofdmTransceiver(request):
    if request.method == 'POST':
        form = ParameterForm(request.POST)
        if form.is_valid():
            input_text = form.cleaned_data['input_text']
            # Get values of subcarrier and mod_order from input text
            n_subcarriers = 0
            n_pilot = 0
            pilot_value = 0+0j
            cycle_prefix = 0
            snr_db = 0
            bits_per_symbol = 0

            #assign module variable with 0
            allCarriers = None 
            pilotCarriers = None
            dataCarriers = None
            payloadBits_per_OFDM = None
            bits = None
            bits_paralel = None
            const_mapp = None
            const_demapp = None
            ofdm_data = None
            ofdm_time = None
            ofdm_tx = None 
            H_exact = None
            ofdm_rx = None
            ofdm_rx_cpremove = None
            ofdm_freq = None
            hest_at_pilots = None
            hest = None
            equalized_hest = None
            receiced_const = None
            bits_received = None
            hard_decision = None
            bits_serial = None

            #assign imagepath with 0
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
                        n_subcarriers = int(param.split('=')[1])
                    elif param.startswith('np='):
                        n_pilot = int(param.split('=')[1])
                    elif param.startswith('pv='):
                        pilot_value = complex(param.split('=')[1])
                    elif param.startswith('bps='):
                        bits_per_symbol = int(param.split('=')[1])
                    elif param.startswith('cp='):
                        cycle_prefix = int(param.split('=')[1])
                    elif param.startswith('snr='):
                        snr_db = int(param.split('=')[1])

                    #Fungsi menampilkan Plot   
                    elif param.startswith('bit-generate'):
                        allCarriers,pilotCarriers, dataCarriers, payloadBits_per_OFDM = m.subcarrier_config(n_subcarriers, n_pilot, bits_per_symbol)
                        bits = m.bit_generator(payloadBits_per_OFDM)
                        subcarriers_img_base64 = p.get_plot_subcarriers(pilotCarriers, dataCarriers, n_subcarriers)
                        bit_generator_img_base64 = p.get_plot_bitgenerate(bits)

                    elif param.startswith('tx-serial-parallel'):
                        bits_paralel = m.serial_to_paralel(bits, dataCarriers, bits_per_symbol)

                    elif param.startswith('digital-modulation'):
                        const_mapp, const_demapp, mapping_table  = m.Mapping(bits_paralel, bits_per_symbol)
                        constellation_map_img_base64 = p.get_plot_constellation_map(const_mapp, mapping_table,bits_per_symbol)

                    elif param.startswith('ifft'):
                        ofdm_data = m.OFDM_symbol(n_subcarriers, const_mapp, pilotCarriers, pilot_value, dataCarriers)
                        ofdm_time = m.IDFT(ofdm_data)
                        ifft_output_img_base64 = p.get_plot_ifft_output(ofdm_time)

                    elif param.startswith('add-cp'):
                        ofdm_tx = m.addCP(ofdm_time, cycle_prefix)
                    
                    elif param.startswith('ofdm-channel'):
                        H_exact, ofdm_rx = m.channel(n_subcarriers, ofdm_tx, snr_db)
                        channel_response_img_base64 =p.get_plot_channel_response(allCarriers,H_exact, n_subcarriers)
                    
                    elif param.startswith('awgn'):
                        ofdm_txrx_img_base64 = p.get_plot_ofdm_txrx(ofdm_tx, ofdm_rx)

                    elif param.startswith('remove-cp'):
                        ofdm_rx_cpremove = m.removeCP(ofdm_rx, cycle_prefix, n_subcarriers)
                    
                    elif param.startswith('fft'):
                        ofdm_freq = m.DFT(ofdm_rx_cpremove)

                    elif param.startswith('equalizer'):
                        hest_at_pilots, hest = m.channelEstimate(ofdm_freq, pilotCarriers, pilot_value, allCarriers)
                        channel_estimation_img_base64 = p.get_plot_channel_estimation(allCarriers, H_exact, pilotCarriers, hest_at_pilots, hest)
                        equalized_hest = m.equalize(ofdm_freq, hest)
                        receiced_const = m.get_payload(equalized_hest, dataCarriers)
                        received_const_img_base64 = p.get_plot_received_const(receiced_const)
                        bits_received, hard_decision = m.Demapping(receiced_const, const_demapp)
                        equalizer_const_img_base64 = p.get_plot_equalizer_const(receiced_const, hard_decision)
                    
                    elif param.startswith('rx-parallel-serial'):
                        bits_serial = m.PS(bits_received)

                    
                    elif param.startswith('received-bits'):
                        received_bits_img_base64 = p.get_plot_received_bits(bits_serial, bits)
                    
                except Exception as e:
                    print(e)
                    error_message = "Kode yang anda masukkan salah"
                
            print (n_subcarriers, n_pilot,pilot_value,bits_per_symbol,cycle_prefix,snr_db)
            #OFDM Function
            """
            allCarriers,pilotCarriers, dataCarriers, payloadBits_per_OFDM = m.subcarrier_config(n_subcarriers, n_pilot, bits_per_symbol)
            bits = m.bit_generator(payloadBits_per_OFDM)
            subcarriers_img_base64 = p.get_plot_subcarriers(pilotCarriers, dataCarriers, n_subcarriers)
            bit_generator_img_base64 = p.get_plot_bitgenerate(bits)            
            
            bits_paralel = m.serial_to_paralel(bits, dataCarriers, bits_per_symbol)
            const_mapp, const_demapp  = m.Mapping(bits_paralel, bits_per_symbol)
            constellation_map_img_base64 = p.get_plot_constellation_map(const_mapp)

            ofdm_data = m.OFDM_symbol(n_subcarriers, const_mapp, pilotCarriers, pilot_value, dataCarriers)
            ofdm_time = m.IDFT(ofdm_data)
            ifft_output_img_base64 = p.get_plot_ifft_output(ofdm_time)

            ofdm_tx = m.addCP(ofdm_time, cycle_prefix)
            H_exact, ofdm_rx = m.channel(n_subcarriers, ofdm_tx, snr_db)
            channel_response_img_base64 =p.get_plot_channel_response(allCarriers,H_exact, n_subcarriers)
            ofdm_txrx_img_base64 = p.get_plot_ofdm_txrx(ofdm_tx, ofdm_rx)

            ofdm_rx_cpremove = m.removeCP(ofdm_rx, cycle_prefix, n_subcarriers)
            ofdm_freq = m.DFT(ofdm_rx_cpremove)
            hest_at_pilots, hest = m.channelEstimate(ofdm_freq, pilotCarriers, pilot_value, allCarriers)
            channel_estimation_img_base64 = p.get_plot_channel_estimation(allCarriers, H_exact, pilotCarriers, hest_at_pilots, hest)

            equalized_hest = m.equalize(ofdm_freq, hest)
            receiced_const = m.get_payload(equalized_hest, dataCarriers)
            received_const_img_base64 = p.get_plot_received_const(receiced_const)

            bits_received, hard_decision = m.Demapping(receiced_const, const_demapp)
            equalizer_const_img_base64 = p.get_plot_equalizer_const(receiced_const, hard_decision)

            bits_serial = m.PS(bits_received)
            received_bits_img_base64 = p.get_plot_received_bits(bits_serial)
            """
        
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

#Backup Code
'''
def ofdmTransmitter(request):
    input_ofdm = request.POST.get('inputOfdmTransmitter')
    data_ofdm = input_ofdm.split()
    try:
        kata,show = m.inputData(data_ofdm)
        txt_signal = {
            'ofdmtOutput' : show,
            'active_tab':'nav-ofdmt-tab'}
        if kata is False:
            raise ValueError('Salah memasukkan kata dalam kode')
    except Exception as e:
        print(e)
        txt_signal = {
            'typoCodeOfdmTransmitter': e,
            'active_tab':'nav-ofdmt-tab'}
    return render(request, 'ofdmsim/ofdmsim.html', dict(ChainMap(txt_signal,txt)))
'''

