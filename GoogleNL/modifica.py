import argparse
import sys
import pandas as pd
import numpy as np
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import six
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import math

fIn = open('googleSentiment.csv', 'r')
fOut = open('googleSentimentMod.csv', 'w')

for line in fIn:
	fOut.write('"'+line[:19] + '" ; ' + line[20:]+'\n')


    
fIn.close()
fOut.close()
