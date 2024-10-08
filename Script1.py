import pandas as pd


def update_w_b(time, temps, w, b, alpha):
    dw = 0.0
    db = 0.0
    N = len(time)

    for i in range(N):
        dw += -2 * time[i] * (temps[i] - ((time[i] * w) + b))
        db += -2 * (temps[i] - ((time[i] * w) + b))

    w = w - (1 / float(N)) * dw * alpha
    b = b - (1 / float(N)) * db * alpha

    return w, b


def train(time, temps, w, b, alpha, epochs):
    for e in range(epochs):
        w, b = update_w_b(time, temps, w, b, alpha)

        if e % 400 == 0:
            print("epoch: ", e, "loss: ", avg_loss(time, temps, w, b))
    return w, b


def avg_loss(time, temps, w, b):
    N = len(time)
    total_error = 0.0

    for i in range(N):
        total_error += (temps[i] - ((time[i] * w) + b)) ** 2

    return total_error / float(N)


def predict(w, b, time):
    return w * time + b


# Normalize the time data
time = [2000, 2001, 2002, 2003, 2004]
temps = [24, 23, 26, 28, 25]
time_min = min(time)
time_max = max(time)
time_norm = [(t - time_min) / (time_max - time_min) for t in time]

w = 0.5
b = 0.3
alpha = 0.001  # Use a smaller learning rate

w, b = train(time_norm, temps, w, b, alpha, 1200)

# Predict the value for 2005
time_pred = (2005 - time_min) / (time_max - time_min)
print(predict(w, b, time_pred))
