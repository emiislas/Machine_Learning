#Numpy only discrete fourier transform method and example
import numpy as np
import matplotlib.pyplot as plt

#We start with a time-series input signal
def f(t):
    #2 harmonic signals f(t) = sin(2t) + sin(4t)
    return np.sin(2*t) + np.sin(4*t)
    
def dft(y):
    N = len(y)
    y_hat = np.zeros(N, dtype=complex)
    for k in range(N):
        y_hat[k] = np.sum([y[n]*np.exp(-1j*2*np.pi*(k/N)*n) for n in range(N)])
    return y_hat

def idft(y_hat):
    N = len(y_hat)
    i_y_hat = np.zeros(N, dtype=complex)
    for n in range(N):
        i_y_hat[n] = (1/N)*np.sum([y_hat[k]*np.exp(1j*2*np.pi*(k/N)*n) for k in range(N)])

    return i_y_hat

# t \in [-4,4], \Delta_t = 0,05, |T| = 1610 
dt = 0.005
t = np.arange(-4,4.05,dt)

y_hat = dft(f(t))

N = len(t)
freqs = np.arange(N) / (N * dt)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(t, f(t))
axes[0].set_title("Time Domain")
axes[0].set_xlabel("t")

axes[1].stem(freqs[:N//2], np.abs(y_hat[:N//2]) / N, markerfmt=" ")
axes[1].set_title("Frequency Domain (magnitude)")
axes[1].set_xlabel("Frequency (Hz)")

y_reconstructed = idft(y_hat)

print("Reconstruction error:", np.max(np.abs(f(t) - y_reconstructed)))

plt.tight_layout()
plt.show()
