import csv
import os
import re

path = "/vol/bitbucket/jhc4318/fyp"
log = "log_unet_refine_gaussian1d_10/log_all_2022_06_20_12_44_57.log"
losses = "loss1d_10.csv"

with open(os.path.join(path, losses), "w") as data:
  writer = csv.writer(data)
  writer.writerow(["epoch", "step", "d_loss", "g_loss", "g_perceptual_loss", "g_mse", "g_freq"])

  with open(os.path.join(path, log), "r") as output:
    for line in output:
      if not line.startswith("Epoch["):
        continue

      values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[:-1]
      del values[1]
      row = [float(x) for x in values]
      row[0] = int(row[0])
      row[1] = int(row[1])
      writer.writerow(row)

print("Success")