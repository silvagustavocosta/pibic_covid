from cmath import nan
from re import X
import string
from tracemalloc import start
from turtle import clear
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from os import listdir, system
from os.path import isfile, join
from os.path import exists
import matplotlib
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pyplot import figure
import os
import shutil
import csv


folder_path = "/mnt/d/PIBIC/Data Final"

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        # Rename the file to change the extension from .txt to .csv
        os.rename(os.path.join(folder_path, filename),
                  os.path.join(folder_path, filename[:-4] + ".csv"))
