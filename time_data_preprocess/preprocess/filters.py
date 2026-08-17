from scipy.signal import butter, filtfilt


def bandpass_filter(data, lowcut=0.5, highcut=45, fs=128, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)
