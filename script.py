import numpy as np
import matplotlib.pyplot as plt
SAMPLE_RATE = 44100
BIT_DURATION = 1.0
FREQ_LOW = 440
FREQ_HIGH = 880

def generate_tone(frequency, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    window = np.hanning(len(tone))
    return tone * window

def encode_nrz(data_bits):
    audio_signal = np.array([])
    for bit in data_bits:
        freq = FREQ_HIGH if bit == '1' else FREQ_LOW
        tone = generate_tone(freq, BIT_DURATION)
        audio_signal = np.concatenate([audio_signal, tone])
    return audio_signal

def encode_manchester(data_bits):
    audio_signal = np.array([])
    for bit in data_bits:
        if bit == '1':
            tone1 = generate_tone(FREQ_HIGH, BIT_DURATION/2)
            tone2 = generate_tone(FREQ_LOW, BIT_DURATION/2)
        else:
            tone1 = generate_tone(FREQ_LOW, BIT_DURATION/2)
            tone2 = generate_tone(FREQ_HIGH, BIT_DURATION/2)
        bit_signal = np.concatenate([tone1, tone2])
        audio_signal = np.concatenate([audio_signal, bit_signal])
    return audio_signal

def detect_frequency(audio_segment, sample_rate=SAMPLE_RATE):
    seg = audio_segment - np.mean(audio_segment)
    seg = seg * np.hanning(len(seg))
    fft = np.fft.fft(seg)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    magnitude = np.abs(fft[:len(fft)//2])
    freqs_positive = freqs[:len(freqs)//2]
    peak_idx = np.argmax(magnitude)
    detected_freq = abs(freqs_positive[peak_idx])
    return detected_freq

def frequency_to_bit(frequency, threshold=660):
    return '1' if frequency > threshold else '0'

def decode_nrz(audio_signal, num_bits, sample_rate=SAMPLE_RATE):
    samples_per_bit = int(sample_rate * BIT_DURATION)
    decoded_bits = ""
    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit
        if end_idx > len(audio_signal):
            break
        mid_start = start_idx + samples_per_bit // 4
        mid_end = end_idx - samples_per_bit // 4
        segment = audio_signal[mid_start:mid_end]
        freq = detect_frequency(segment, sample_rate)
        bit = frequency_to_bit(freq)
        decoded_bits += bit
    return decoded_bits

def decode_manchester(audio_signal, num_bits, sample_rate=SAMPLE_RATE):
    samples_per_bit = int(sample_rate * BIT_DURATION)
    decoded_bits = ""
    for i in range(num_bits):
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit
        if end_idx > len(audio_signal):
            break
        mid_point = start_idx + samples_per_bit // 2
        first_half = audio_signal[start_idx + samples_per_bit//8 : mid_point - samples_per_bit//8]
        second_half = audio_signal[mid_point + samples_per_bit//8 : end_idx - samples_per_bit//8]
        freq1 = detect_frequency(first_half, sample_rate)
        freq2 = detect_frequency(second_half, sample_rate)
        state1 = frequency_to_bit(freq1)
        state2 = frequency_to_bit(freq2)
        if state1 == '1' and state2 == '0':
            bit = '1'
        elif state1 == '0' and state2 == '1':
            bit = '0'
        else:
            bit = '?'
        decoded_bits += bit
    return decoded_bits

def adicionar_ruido(audio_signal, snr_db):
    signal_power = np.mean(audio_signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear if snr_linear>0 else signal_power*1e12
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_signal))
    return audio_signal + noise

original_bits = "10110"
num_bits = len(original_bits)
nrz_clean = encode_nrz(original_bits)
manchester_clean = encode_manchester(original_bits)

snr_values = list(range(20, -61, -2))
trials = 8
nrz_errors = []
manchester_errors = []

for snr in snr_values:
    ne = 0.0
    me = 0.0
    for _ in range(trials):
        noisy_nrz = adicionar_ruido(nrz_clean, snr)
        noisy_man = adicionar_ruido(manchester_clean, snr)
        decoded_nrz = decode_nrz(noisy_nrz, num_bits)
        decoded_man = decode_manchester(noisy_man, num_bits)
        ne += sum(1 for a,b in zip(decoded_nrz, original_bits) if a!=b) + (num_bits - len(decoded_nrz))
        me += sum(1 for a,b in zip(decoded_man, original_bits) if a!=b) + (num_bits - len(decoded_man))
    nrz_errors.append(ne/trials)
    manchester_errors.append(me/trials)

def find_thresholds(snr_vals, errors, num_bits):
    first_err = None
    first_all = None
    for snr, err in zip(snr_vals, errors):
        if first_err is None and err > 0:
            first_err = snr
        if first_all is None and err >= num_bits:
            first_all = snr
    return first_err, first_all

nrz_first_error, nrz_first_all = find_thresholds(snr_values, nrz_errors, num_bits)
man_first_error, man_first_all = find_thresholds(snr_values, manchester_errors, num_bits)

plt.figure(figsize=(8,4))
plt.plot(snr_values, nrz_errors)
plt.title('NRZ: SNR vs Número médio de erros (trials={})'.format(trials))
plt.xlabel('SNR (dB)')
plt.ylabel('Erros médios por {} bits'.format(num_bits))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(snr_values, manchester_errors)
plt.title('Manchester: SNR vs Número médio de erros (trials={})'.format(trials))
plt.xlabel('SNR (dB)')
plt.ylabel('Erros médios por {} bits'.format(num_bits))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("NRZ mean errors:", [round(v,2) for v in nrz_errors])
print("Manchester mean errors:", [round(v,2) for v in manchester_errors])
print(f"NRZ thresholds: first error at {nrz_first_error}, all bits at {nrz_first_all}")
print(f"Manchester thresholds: first error at {man_first_error}, all bits at {man_first_all}")