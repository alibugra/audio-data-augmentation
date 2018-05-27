import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioAugmentation:
    def read_audio_file(self, file_path):
        input_length = 16000
        data = librosa.core.load(file_path)[0]
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def write_audio_file(self, file, data, sample_rate=16000):
        librosa.output.write_wav(file, data, sample_rate)

    def plot_time_series(self, data):
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()

    def add_noise(self, data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def shift(self, data):
        return np.roll(data, 1600)

    def stretch(self, data, rate=1):
        input_length = 16000
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

# Create a new instance from AudioAugmentation class
aa = AudioAugmentation()

# Read and show cat sound
data = aa.read_audio_file("data/cat.wav")
aa.plot_time_series(data)

# Adding noise to sound
data_noise = aa.add_noise(data)
aa.plot_time_series(data_noise)

# Shifting the sound
data_roll = aa.shift(data)
aa.plot_time_series(data_roll)

# Stretching the sound
data_stretch = aa.stretch(data, 0.8)
aa.plot_time_series(data_stretch)

# Write generated cat sounds
aa.write_audio_file('output/generated_cat1.wav', data_noise)
aa.write_audio_file('output/generated_cat2.wav', data_roll)
aa.write_audio_file('output/generated_cat3.wav', data_stretch)
