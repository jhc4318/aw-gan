import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

path = "/vol/bitbucket/jhc4318/fyp"
losses = "loss1d_50.csv"

df = pd.read_csv(os.path.join(path, losses))
fig, ax = plt.subplots()

y1 = "g_perceptual_loss"
y2 = "g_mse"
ax.plot(df.index, df[y1], color="red", linewidth=0.8)
ax.set_xlabel("Step")
ax.set_ylabel(y1, color="red")
# ax.set_ylim([1, 2.6])

ax2 = ax.twinx()
ax2.plot(df.index, df[y2], color="blue", linestyle='--', linewidth=0.8)
ax2.set_ylabel(y2, color="blue")
# ax2.set_ylim([0.009, 0.018])

# ax.set_xlim([800, 900])
fig.savefig("test.png")