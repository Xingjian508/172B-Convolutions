import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import scipy.fftpack as sp
from scipy.stats import kurtosis, skew, mode, iqr, gmean, hmean, variation, trim_mean as tstd
from scipy.spatial.distance import euclidean as gstd
from scipy.stats import median_abs_deviation, entropy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

class StatFeatExtractor:
  def __init__(self, file_path):
    self.path = file_path

  def extract_feats(self):
    og_features = self.process_audio_file(self.path)
    feature_names = ["Mode", "Kurtosis", "Skewness", "Mean", "IQR", "Geometric Mean",
                         "Harmonic Mean", "MAD", "COV", "Variance", "Standard Deviation",
                         "Generalized STD", "Entropy"]
    features_dict = dict()
    for i, feature_name in enumerate(feature_names):
      features_dict[feature_name] = [og_features[i, 0] for i in range(184)]

    energy, zcr, pitch = self.process_and_plot_audio(self.path)
    features_dict['Energy'] = energy
    features_dict['ZCR'] = zcr
    features_dict['Pitch'] = pitch
    return features_dict
    
  def extract_statistical_features(self, signal):
    ft = sp.fft(signal)
    magnitude = np.abs(ft)
    spec = magnitude
  
    k = kurtosis(spec)
    s = skew(spec)
    mean = np.mean(spec)
    z = np.array(mode(spec)[0])
    mode_var = float(z)
    i = iqr(spec)
    g = gmean(spec)
    h = hmean(spec)
    dev = median_abs_deviation(spec)
    var = variation(spec)
    variance = np.var(spec)
    std = tstd(spec, 0.1)
  
    gstd_var = 0
    try:
      gstd_var = gstd(spec)
    except:
      pass
      
    ent = entropy(spec)
  
    features = [mode_var, k, s, mean, i, g, h, dev, var, variance, std, gstd_var, ent]
    features = normalize([features])
    features = np.array(features)
    features = np.reshape(features, (13,))
  
    return features
  
  def process_audio_file(self, file_path, frame_length=2048, hop_length=512):
    sr, audio = wav.read(file_path)
    if audio.ndim > 1:
      audio = np.mean(audio, axis=1)
  
    num_frames = (len(audio) - frame_length) // hop_length + 1
    all_features = np.zeros((num_frames, 13))
  
    for i in range(num_frames):
      start = i * hop_length
      end = start + frame_length
      frame = audio[start:end]
      all_features[i, :] = self.extract_statistical_features(frame)
  
    # plt.figure(figsize=(15, 20))
    # feature_names = ["Mode", "Kurtosis", "Skewness", "Mean", "IQR", "Geometric Mean",
    #          "Harmonic Mean", "MAD", "COV", "Variance", "Standard Deviation",
    #          "Generalized STD", "Entropy"]
    # for idx, name in enumerate(feature_names):
    #   plt.subplot(len(feature_names), 1, idx+1)
    #   plt.plot(all_features[:, idx])
    #   plt.title(name)
    # plt.tight_layout()
    # plt.show()
  
    return all_features
  
  def frame_energy(self, signal, frame_size, hop_size):
    """Calculate the energy of the signal in each frame."""
    return np.array([
      np.sum(np.square(signal[i:i+frame_size]))
      for i in range(0, len(signal) - frame_size + 1, hop_size)
    ])
  
  def frame_zcr(self, signal, frame_size, hop_size):
    """Calculate the zero-crossing rate of the signal in each frame."""
    return np.array([
      np.mean(np.abs(np.diff(np.sign(signal[i:i+frame_size])))) / 2
      for i in range(0, len(signal) - frame_size + 1, hop_size)
    ])
  
  def autocorrelation_pitch(self, signal, sr, frame_length, hop_length, fmin=50, fmax=500):
    """Estimate pitch using autocorrelation."""
    pitch_frequencies = []
    for i in range(0, len(signal) - frame_length, hop_length):
      frame = signal[i:i+frame_length]
      autocorr = np.correlate(frame, frame, mode='full')
      mid = len(autocorr) // 2
      autocorr = autocorr[mid:]
  
      d = np.diff(autocorr)
      start = np.nonzero(d > 0)[0][0]
      peak_idx = np.argmax(autocorr[start:]) + start
      pitch_period = peak_idx if peak_idx != start else 0
      pitch_freq = sr / pitch_period if pitch_period != 0 else 0
  
      if fmin <= pitch_freq <= fmax:
        pitch_frequencies.append(pitch_freq)
      else:
        pitch_frequencies.append(0) 
  
    return np.array(pitch_frequencies)
  
  def process_and_plot_audio(self, file_path):
    """Process and plot the energy, ZCR, and pitch frequency of an audio file."""
    sr, y = wav.read(file_path)
    if y.ndim > 1:
      y = np.mean(y, axis=1)
    y = y.astype(np.float32) / np.max(np.abs(y))
  
    frame_length = 2048
    hop_length = 512
  
    energy_frames = self.frame_energy(y, frame_length, hop_length)
    zcr_frames = self.frame_zcr(y, frame_length, hop_length)
    pitch_frames = self.autocorrelation_pitch(y, sr, frame_length, hop_length)
  
    # plt.figure(figsize=(18, 10))
    # plt.subplot(3, 1, 1)
    # plt.plot(energy_frames)
    # plt.title('Energy per Frame')
    # plt.ylabel('Energy')
  
    # plt.subplot(3, 1, 2)
    # plt.plot(zcr_frames)
    # plt.title('Zero-Crossing Rate per Frame')
    # plt.ylabel('ZCR')
  
    # plt.subplot(3, 1, 3)
    # plt.plot(pitch_frames)
    # plt.title('Pitch Frequency per Frame')
    # plt.xlabel('Frame')
    # plt.ylabel('Frequency (Hz)')
  
    # plt.tight_layout()
    # plt.show()
    return energy_frames, zcr_frames, pitch_frames

