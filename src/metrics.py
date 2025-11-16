import pandas as pd
import numpy as np
import metrics as M
from math import sqrt
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
from scipy.stats import ttest_ind

df = pd.read_csv("../data/health_study_dataset.csv")


def statistik(df):
    statistics = df[["age", "weight", "height", "systolic_bp", "cholesterol"]].agg(["mean", "median", "min", "max"])
    return statistics
def sick_ones(df):
    sick = df["disease"] == 1
    counts = sick.sum()
    return counts
def disease_sim(df):
    counts = M.sick_ones(df)
    people_quantity = len(df)
    disease_count = counts / people_quantity
    np.random.seed(42)
    sim_dis = np.random.choice([0, 1], size=1000, p=[1 - disease_count, disease_count])
    simulated_disease = sim_dis.mean()
    return simulated_disease, sim_dis, disease_count,people_quantity
def true_mean(df):
    np.random.seed(123)
    systolic_bp = df["systolic_bp"]
    true_mean = float(np.mean(systolic_bp))
    return true_mean


def ci_systolic_bp(df):
    data = df["systolic_bp"].dropna()
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)

    z = 1.96
    
    half_width = z * std / sqrt(n)

   
    lo, hi = mean - half_width, mean + half_width
    return lo, hi
def ci_systolic_bp_bootstrap(df, B=5000, confidence=0.95):
    data = df["systolic_bp"].dropna()
    n = len(data)
    boot_means = np.empty(B)
    for b in range(B):
        boot_sample = np.random.choice(data, size= n, replace=True)
        boot_means[b] = np.mean(boot_sample)
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(boot_means, [100*alpha, 100*(1-alpha)])
    return float(lo), float(hi), float(np.mean(data))
def hypotesprövning(df):
    rokare = df[df["smoker"] == "Yes"] ["systolic_bp"]
    ickerokare = df[df["smoker"] == "No"] ["systolic_bp"]

    t_stat, p_value = stats.ttest_ind(rokare, ickerokare, equal_var=True)
    t_stat_w, p_value_w = stats.ttest_ind(rokare, ickerokare, equal_var=False)


    return rokare, ickerokare, t_stat, p_value, t_stat_w, p_value_w

