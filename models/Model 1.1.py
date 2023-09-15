# description
"""
Basic RSI + EMA strategy with a hold period. The hold period is the number of observations to wait before closing a
position. This is to prevent the model from opening and closing positions too frequently. The hold period is set to 10
observations by default.
"""

# import libraries
import numpy as np
import pandas as pd
import datetime
import math
import csv
from datetime import datetime
from datetime import timedelta
import pandas_ta as ta
import random

random.seed(1234)
# Set the random seed
seed_value = 1234
np.random.seed(seed_value)
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')



