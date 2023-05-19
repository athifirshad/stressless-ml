from hrvanalysis import get_frequency_domain_features
from hrvanalysis import get_frequency_domain_features
from hrvanalysis import get_sampen
import numpy as np
import scipy.stats as sc
from scipy.signal import welch
import antropy as ant
import csv

bpm = []
nn_intervals = []
with open(r'sample.csv', 'r') as csvfile:
    
    csvreader = csv.reader(csvfile)
    
    for row in csvreader:
        bpm.append(float(row[0]))
        nn_intervals.append(float(row[1]))

    for i in range(0,len(bpm)):
        print(bpm[i])

    for i in range(0,len(bpm)):
        print(nn_intervals[i])


frequency_domain_features = get_frequency_domain_features(nn_intervals)
sampen_n = get_sampen(nn_intervals)


relative_rr = []
vlf=0
lf=0
hf=0
tp=0

def mean_rr(nn_intervals):
  mean_rr = np.mean(nn_intervals)
  return mean_rr

def median_rr(nn_intervals):
  sorted_nn = np.sort(nn_intervals)
  n = len(sorted_nn)
  if n % 2 == 0:
    median_rr = (sorted_nn[int(n / 2) - 1] + sorted_nn[int(n / 2)]) / 2
  else:
    median_rr = sorted_nn[int((n + 1) / 2) - 1]
  return median_rr

def sdrr(nn_intervals):
  diff = np.diff(nn_intervals)
  sdrr = np.sqrt(np.sum(diff**2) / (len(diff) - 1))
  return sdrr

def rmssd(nn_intervals):
  diff = np.diff(nn_intervals)
  rmssd = np.sqrt(np.mean(diff**2))
  return rmssd

def sdsd(nn_intervals):
  diff = np.diff(nn_intervals)
  sdsd = np.sqrt(np.sum(diff**2) / (len(diff) - 1))
  return sdsd

def sdrr_rmssd(nn_intervals):
  sdr = sdrr(nn_intervals)
  rms = rmssd(nn_intervals)
  sdr_rmssd = sdr / rms
  return sdr_rmssd

def hr(bpm):
  mean_rr = np.mean(bpm)
  return mean_rr

def pnn25(nn_intervals):
  nn_diff = np.abs(np.diff(nn_intervals))
  nn_diff_greater_than_25ms = nn_diff[nn_diff > 25]
  pnn25 = float(len(nn_diff_greater_than_25ms)) / float(
    len(nn_intervals)) * 100
  return pnn25

def pnn50(nn_intervals):
  nn_diff = np.abs(np.diff(nn_intervals))
  nn_diff_greater_than_50ms = nn_diff[nn_diff > 50]
  pnn50 = float(len(nn_diff_greater_than_50ms)) / float(
    len(nn_intervals)) * 100
  return pnn50

#def sd1(nn_intervals):
 # diff_nn = np.diff(nn_intervals)
  #sd1 = np.sqrt(2) * np.sqrt(
   # np.var(diff_nn) - 0.5 * (np.var(nn_intervals) + np.var(diff_nn)))
  #return sd1

def sd1(rr_intervals):
    # Calculate the differences between successive RR intervals
    rr_intervals = [rr / 1000 for rr in rr_intervals]
    diffs = np.diff(rr_intervals)
    
    # Calculate SD1 using the formula: SD1 = sqrt((1/2N) * sum(diffs^2))
    sd1 = np.sqrt((1 / (2 * len(diffs))) * np.sum(diffs ** 2))
 
    return sd1

def sd2(nn_intervals):
  diff_nn = np.diff(nn_intervals)
  sd2 = np.sqrt(2 * np.var(nn_intervals) - 0.5 *
                (np.var(nn_intervals) + np.var(diff_nn)))
  return sd2

def kurt(nn_intervals):
  kurt = sc.kurtosis(nn_intervals)
  return kurt

def skew(nn_intervals):
  skew = sc.skew(nn_intervals)
  return skew

def rel_rr(nn_intervals):
  for i in range(0, len(nn_intervals) - 1):
    relative_rr.append(((nn_intervals[i + 1] - nn_intervals[i]) /
                        ((nn_intervals[i] + nn_intervals[i + 1]) / 2)) / 1000)
  return (relative_rr)

rel_array = rel_rr(nn_intervals)

def mean_relrr(rel_array):
  mean_relrr = np.mean(rel_array)
  return mean_relrr

def median_relrr(rel_array):
  sorted_nn = np.sort(rel_array)
  n = len(sorted_nn)
  if n % 2 == 0:
    median_relrr = (sorted_nn[int(n / 2) - 1] + sorted_nn[int(n / 2)]) / 2
  else:
    median_relrr = sorted_nn[int((n + 1) / 2) - 1]
  return median_relrr

def sdrr_rel_rr(rel_array):
  diff = np.diff(rel_array)
  sdrr_rel = np.sqrt(np.sum(diff**2) / (len(diff) - 1))
  return sdrr_rel

def rmssd_rel_rr():
  diff = np.diff(rel_array)
  rmssd_rel = np.sqrt(np.mean(diff**2))
  return rmssd_rel

def sdsd_rel_rr(rel_array):
  diff = np.diff(rel_array)
  sdsd = np.sqrt(np.sum(diff**2) / (len(diff) - 1))
  return sdsd

def sdrr_rmssd_rel(sdrr_rel, rmssd_rel):
  return (sdrr_rel / rmssd_rel)

def kurt_rel(rel_array):
  kurt_rel = sc.kurtosis(rel_array)
  return kurt_rel

def skew_rel(rel_array):
  skew_rel = sc.skew(rel_array)
  return skew_rel

def ratiolfhf(lf,hf):
  return (lf/hf)

def ratiohflf(lf,hf):
    return(hf/lf)

def calculate_psd(nn_intervals):
    # Compute PSD using Welch's method
    f, psd = welch(nn_intervals, fs=4.0, nperseg=len(nn_intervals))
    return f, psd

def calculate_power(f, psd):
    # Integrate PSD over desired frequency ranges
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f <= 0.4)
    total_mask = (f >= 0.0033) & (f <= 0.4)

    lf_power = np.trapz(psd[lf_mask], f[lf_mask])
    hf_power = np.trapz(psd[hf_mask], f[hf_mask])
    total_power = np.trapz(psd[total_mask], f[total_mask])
    vlf_power = total_power - hf_power

    return lf_power, hf_power, vlf_power, total_power

def calculate_hf_pct(lf_power, hf_power, vlf_power):
    return (hf_power / (lf_power + vlf_power))

def calculate_lf_pct(lf_power, hf_power):
    return (lf_power / (lf_power + hf_power)) * 100

def calculate_vlf_pct(vlf_power, total_power):
    return (vlf_power / total_power) * 100

def higuchi_fractal_dimension(rr_intervals):
    return ant.higuchi_fd(rr_intervals)


f, psd = calculate_psd(nn_intervals)
lf_power, hf_power, vlf_power, total_power = calculate_power(f, psd)

MEAN_RR = mean_rr(nn_intervals)
MEDIAN_RR = median_rr(nn_intervals)
SDRR = sdrr(nn_intervals)
RMSSD =  rmssd(nn_intervals)
SDSD = sdsd(nn_intervals)
SDRR_RMSSD = sdrr_rmssd(nn_intervals)
HR =  hr(bpm)
pNN25 = pnn25(nn_intervals)
pNN50 = pnn50(nn_intervals)
SD1 = sd1(nn_intervals)
SD2 = sd2(nn_intervals)
KURT = kurt(nn_intervals)
SKEW = skew(nn_intervals)
MEAN_REL_RR = mean_rr(rel_array)
MEDIAN_REL_RR = median_relrr(rel_array)
SDRR_REL_RR = sdrr_rel_rr(rel_array)
RMSSD_REL_RR =  rmssd_rel_rr()
SDSD_REL_RR = sdsd_rel_rr(rel_array)
SDRR_RMSSD_REL_RR = sdrr_rmssd_rel(SDRR_REL_RR, RMSSD_REL_RR)
KURT_REL_RR = kurt_rel(rel_array)
SKEW_REL_RR = skew_rel(rel_array)
VLF = frequency_domain_features["vlf"]
VLF_PCT = calculate_vlf_pct(vlf_power, total_power)
LF = frequency_domain_features["lf"]
LF_PCT = calculate_lf_pct(lf_power, hf_power)
LF_NU = frequency_domain_features["lfnu"]
HF =  frequency_domain_features["hf"]
HF_PCT = calculate_hf_pct(lf_power, hf_power, vlf_power)
HF_NU = frequency_domain_features["hfnu"]
TP = frequency_domain_features["total_power"]
LF_HF = ratiolfhf(LF,HF)
HF_LF = ratiohflf(LF,HF)
sampen = sampen_n["sampen"]
higuci = higuchi_fractal_dimension(nn_intervals)

input = [MEAN_RR,MEDIAN_RR,SDRR,RMSSD,SDSD,SDRR_RMSSD,HR,pNN25,pNN50,SD1,SD2,KURT,SKEW,MEAN_REL_RR,MEDIAN_REL_RR,SDRR_REL_RR,RMSSD_REL_RR,SDSD_REL_RR,SDRR_RMSSD_REL_RR,KURT_REL_RR,SKEW_REL_RR,VLF,VLF_PCT,LF,LF_PCT,LF_NU,HF,HF_PCT,HF_NU,TP,LF_HF,HF_LF,sampen,higuci]

for i in range(0,len(input)):
    print(input[i])

with open('TEST1.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write each integer to a new row in the CSV file
    for integer in input:
        writer.writerow([integer])