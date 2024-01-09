import base64
import random
import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('agg')


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"


def get_plot_bitgenerate(bits, bps):
    plt.figure(figsize=(12, 2))
    # Plot untuk Signal Generator\
    if len(bits) <= 32:
        plt.step(np.arange(0, len(bits)), bits, 'r', linewidth=1, where='post')
        plt.vlines(np.arange(0, len(bits), bps), ymin=0.0, ymax=1.0, color='k',
                   linewidth=1, linestyle="--", zorder=3, label='bits per symbol mark')
        for tbit, bit in enumerate(bits):
            plt.text(tbit + 0.5, 0.5, str(bit))
    else:
        plt.step(np.arange(0, len(bits)), bits, 'r', where='post')
    # plt.ylim([-1,1.5])
    plt.legend(fontsize=10)
    plt.title('Output Bit Generator')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    bit_generator_path = 'Simulation/static/files/bitgenerator.png'
    plt.savefig(bit_generator_path)
    bit_generator_img_base64 = get_base64_encoded_image(bit_generator_path)
    plt.clf()
    return bit_generator_img_base64


def get_plot_subcarriers(pilotCarriers, dataCarriers, n_subcarriers):
    plt.figure(figsize=(12, 2))
    plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
    plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
    plt.title('Carriers Configuration')
    plt.legend(fontsize=10, ncol=2)
    plt.xlim((-1, n_subcarriers))
    plt.ylim((-0.1, 0.3))
    plt.xlabel('Carrier index')
    plt.yticks([])
    plt.grid(True)
    subcarriers_path = 'Simulation/static/files/subcarriers.png'
    plt.savefig(subcarriers_path)
    subcarriers_img_base64 = get_base64_encoded_image(subcarriers_path)
    plt.clf()
    return subcarriers_img_base64


def get_plot_channel_response(allCarriers, H_exact, n_subcarriers):
    plt.figure(figsize=(12, 4))
    plt.plot(allCarriers, abs(H_exact))
    plt.title('Channel Corresponding Frequency Response')
    plt.xlabel('Subcarrier index')
    plt.ylabel('$|H(f)|$')
    plt.grid(True)
    plt.xlim(0, n_subcarriers-1)
    channel_response_path = 'Simulation/static/files/channelresponse.png'
    plt.savefig(channel_response_path)
    channel_response_img_base64 = get_base64_encoded_image(
        channel_response_path)
    plt.clf()
    return channel_response_img_base64


def get_plot_constellation_map(const_mapp, mapping_table, bits_per_symbol):

    x1 = np.arange(-4.5, 4.5, 0.1)
    x2 = np.zeros_like(x1)

    y2 = np.arange(-4.5, 4.5, 0.1)
    y1 = np.zeros_like(y2)

    symbolIn = np.array(const_mapp)
    symbol_r = symbolIn.real
    symbol_i = symbolIn.imag
    plt.figure(figsize=(6, 6))
    plt.scatter(symbol_r, symbol_i)  # 实部 虚部 画星座图 a array 从0到n-1
    plt.plot(x1, x2, color='red')
    plt.plot(y1, y2, color='red')

    if bits_per_symbol == 1:
        for b0 in [0, 1]:
            B = (b0,)
            Q = mapping_table[B]
            plt.text(Q.real, Q.imag+0.2, "".join(str(x)
                     for x in B), ha='center')

    elif bits_per_symbol == 2:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b1, b0)
                Q = mapping_table[B]
                plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')\

    elif bits_per_symbol == 3:
        for b2 in [0, 1]:
            for b1 in [0, 1]:
                for b0 in [0, 1]:
                    B = (b2, b1, b0)
                    Q = mapping_table[B]
                    plt.text(Q.real, Q.imag+0.2, "".join(str(x)
                             for x in B), ha='center')

    elif bits_per_symbol == 4:
        for b3 in [0, 1]:
            for b2 in [0, 1]:
                for b1 in [0, 1]:
                    for b0 in [0, 1]:
                        B = (b3, b2, b1, b0)
                        Q = mapping_table[B]
                        plt.text(Q.real, Q.imag+0.2, "".join(str(x)
                                 for x in B), ha='center')

    plt.xlabel('real')
    plt.ylabel('imag')
    plt.title('Constelation Mapp')
    plt.grid(True)
    constellation_map_path = 'Simulation/static/files/constellationmap.png'
    plt.savefig(constellation_map_path)
    constellation_map_img_base64 = get_base64_encoded_image(
        constellation_map_path)
    plt.clf()
    return constellation_map_img_base64


def get_plot_ifft_output(ofdm_time):
    plt.figure(figsize=(12, 4))
    plt.plot(abs(ofdm_time), label='TX signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time')
    plt.ylabel('$|x(t)|$')
    plt.title('OFDM Signal $(IFFT Output)$')
    plt.grid(True)
    ifft_output_path = 'Simulation/static/files/ifftoutput.png'
    plt.savefig(ifft_output_path)
    ifft_output_img_base64 = get_base64_encoded_image(ifft_output_path)
    plt.clf()
    return ifft_output_img_base64


def get_plot_ofdm_txrx(ofdm_tx, ofdm_rx):
    plt.figure(figsize=(12, 4))
    plt.plot(abs(ofdm_tx), label='TX signal with CP')
    plt.plot(abs(ofdm_rx), label='RX signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time')
    plt.ylabel('$|x(t)|$')
    plt.title('OFDM Signal $(TX Signal With CP & RX Signal)$')
    plt.grid(True)
    ofdm_txrx_path = 'Simulation/static/files/ofdmtxrx.png'
    plt.savefig(ofdm_txrx_path)
    ofdm_txrx_img_base64 = get_base64_encoded_image(ofdm_txrx_path)
    plt.clf()
    return ofdm_txrx_img_base64


def get_plot_channel_estimation(allCarriers, H_exact, pilotCarriers, hest_at_pilots, hest):
    plt.figure(figsize=(12, 4))
    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(hest),
             label='Estimated channel via interpolation')
    plt.grid(True)
    plt.xlabel('Carrier index')
    plt.ylabel('$|H(f)|$')
    plt.legend(fontsize=10)
    plt.ylim(0, 2)
    channel_estimation_path = 'Simulation/static/files/channelestimation.png'
    plt.savefig(channel_estimation_path)
    channel_estimation_img_base64 = get_base64_encoded_image(
        channel_estimation_path)
    plt.clf()
    return channel_estimation_img_base64


def get_plot_received_const(received_const):
    plt.figure(figsize=(6, 6))
    plt.plot(received_const.real, received_const.imag, 'bo')
    plt.grid(True)
    # plt.ylim([-4,4])
    # plt.xlim([-4,4])
    plt.xlabel('Real part')
    plt.ylabel('Imaginary Part')
    plt.title("Received constellation")
    received_const_path = 'Simulation/static/files/receivedconst.png'
    plt.savefig(received_const_path)
    received_const_img_base64 = get_base64_encoded_image(received_const_path)
    plt.clf()
    return received_const_img_base64


def get_plot_equalizer_const(received_const, hard_decision, const_demapp, const_mapp):
    plt.figure(figsize=(6, 6))
    y = [const_demapp[C] for C in hard_decision]
    t = [const_demapp[C] for C in const_mapp]
    i = 0
    x1 = np.arange(-4.5, 4.5, 0.1)
    x2 = np.zeros_like(x1)

    y2 = np.arange(-4.5, 4.5, 0.1)
    y1 = np.zeros_like(y2)
    for qam, hard, y, const, t in zip(received_const, hard_decision, y, const_mapp, t):
        plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
        plt.plot(hard_decision.real, hard_decision.imag, 'ro',
                 label='Restored Constellation via Hard Decision'if i == 0 else "")
        plt.scatter(const.real, const.imag, s=100, marker='x', color='g',
                    linewidth=2, label='Transmitted Constelation'if i == 0 else "")
        plt.text(const.real, const.imag+0.2, str(t).replace(',',
                 '').replace('(', '').replace(')', ''), ha='center')
        plt.text(hard.real, hard.imag+0.2, str(y).replace(',',
                 '').replace('(', '').replace(')', ''), ha='center')
        i = i+1
    plt.plot(x1, x2, color='red')
    plt.plot(y1, y2, color='red')
    plt.legend(scatterpoints=1)

    plt.grid(True)
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Hard Decision demapping')

    # plt.ylim([-4,4])
    # plt.xlim([-4,4])
    equalizer_const_path = 'Simulation/static/files/equalizerconst.png'
    plt.savefig(equalizer_const_path)
    equalizer_const_img_base64 = get_base64_encoded_image(equalizer_const_path)
    plt.clf()
    return equalizer_const_img_base64,


def get_plot_received_bits(bits_serial, bits, bps):
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    if len(bits) <= 32:
        plt.step(np.arange(0, len(bits)), bits, 'r', linewidth=1,
                 where='post', label='Transmited Bits')
        plt.vlines(np.arange(0, len(bits), bps), ymin=0.0, ymax=1.0, color='k',
                   linewidth=1, linestyle="--", zorder=3, label='bits per symbol mark')
        for tbit, bit in enumerate(bits):
            plt.text(tbit + 0.5, 0.5, str(bit))
    else:
        plt.step(np.arange(0, len(bits)), bits, 'r', where='post')
    # plt.ylim([-1,1.5])
    plt.legend(fontsize=10)
    plt.title('Transmited Bits')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout(pad=15)

    plt.subplot(2, 1, 2)
    if len(bits_serial) <= 32:
        plt.step(np.arange(0, len(bits_serial)), bits_serial, 'r',
                 linewidth=1, zorder=1, where='post', label='Restored bits serial')
        plt.vlines((np.arange(0, len(bits_serial)) + 0.5), abs(bits -
                   bits_serial), 'b', linewidth=4, zorder=2, label='bit errors')
        plt.vlines(np.arange(0, len(bits_serial), bps), ymin=0.0, ymax=1.0, color='k',
                   linewidth=1, linestyle="--", zorder=3, label='bits per symbol mark')

        for tbit, bit in enumerate(bits_serial):
            plt.text(tbit + 0.5, 0.5, str(bit))
    else:
        plt.step(np.arange(0, len(bits_serial)), bits_serial, 'r',
                 zorder=1, where='post', label='Restored bits serial')
        plt.vlines((np.arange(0, len(bits_serial)) + 0.5), abs(bits -
                   bits_serial), 'b', linewidth=2, zorder=2, label='bit errors')

    plt.legend(fontsize=10)
    plt.title('Received Bits')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=1)
    received_bits_path = 'Simulation/static/files/receivedbits.png'
    plt.savefig(received_bits_path)
    received_bits_img_base64 = get_base64_encoded_image(received_bits_path)
    plt.clf()
    return received_bits_img_base64
