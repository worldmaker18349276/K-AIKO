import numpy
import scipy
import scipy.fftpack
import scipy.signal

# def hz2mel(f):
#     return 2595.0 * numpy.log10(1.0 + f / 700.0)
# 
# def mel2hz(m):
#     return 700.0 * (10.0**(m / 2595.0) - 1.0)

@numpy.vectorize
def hz2mel(f):
    if f < 1000.0:
        return f * 0.015
    else:
        return numpy.log(f) * 14.54507850578556 - 85.4738428315498

@numpy.vectorize
def mel2hz(m):
    if m < 1000.0:
        return m / 0.015
    else:
        return 1000.0 * numpy.exp(0.06875177742094912 * m - 1.0312766613142368)

def mel_freq(fmin, fmax, n_mels, extend=0):
    mel_min = hz2mel(fmin)
    mel_max = hz2mel(fmax)
    m, dm = numpy.linspace(mel_min, mel_max, n_mels, retstep=True, dtype=numpy.float32)
    if extend != 0:
        m_ = mel_max + numpy.arange(1, extend+1, dtype=numpy.float32)*dm
        m = numpy.concatenate((m, m_))
    return mel2hz(m)

def mel_weights(sr, n_fft, n_mels):
    fft_f = numpy.linspace(0, sr/2, n_fft//2+1, dtype=numpy.float32)
    mel_f = mel_freq(0, sr/2, n_mels, 2)

    weights = numpy.empty((n_mels, n_fft//2+1), dtype=numpy.float32)
    for i in range(n_mels):
        mel_df1 = mel_f[i+1] - mel_f[i]
        mel_df2 = mel_f[i+2] - mel_f[i+1]
        mel_df = (mel_f[i+2] - mel_f[i])/2
        lower = (fft_f   - mel_f[i]) / mel_df1
        upper = (mel_f[i+2] - fft_f) / mel_df2
        weights[i] = numpy.maximum(0, numpy.minimum(lower, upper)) / mel_df

    return weights

def power2db(power):
    db = 10.0 * numpy.log10(numpy.maximum(1e-10, power))
    db = numpy.maximum(db, db.max() - 80.0)
    return db

def melspectrogram(sr, win_length, hop_length):
    weights = mel_weights(sr, win_length, 128)

    window = scipy.signal.get_window("hann", win_length)

    x = numpy.zeros(win_length, dtype=numpy.float32)
    while True:
        X = power2db(weights.dot(numpy.abs(numpy.fft.rfft(x*window))**2))
        x[:-hop_length] = x[hop_length:]
        x[-hop_length:] = yield X
