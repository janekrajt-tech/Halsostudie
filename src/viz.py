import matplotlib.pyplot as plt
import metrics as M
import numpy as np
import pandas as pd

df = pd.read_csv("../data/health_study_dataset.csv")

def hist_blodtryck(df):
    systolic_bp = df["systolic_bp"]
    fig, ax = plt.subplots()
    ax.hist(systolic_bp, color="darkred", edgecolor= "black", bins=25)
    mean_bp = systolic_bp.mean()
    plt.axvline(mean_bp, color="black", linestyle="--", linewidth= 2, label=f"Medel: {mean_bp:.1f}" )

    ax.set_title("Histogram över blodtryck")
    ax.set_xlabel("Blodtryck")
    ax.set_ylabel("Antal personer")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def box_sex_weight(df):
    male_weight = df[df["sex"] == "M"]["weight"]
    female_weight = df[df["sex"] == "F"]["weight"]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    box = ax.boxplot([male_weight, female_weight], labels=["Male", "Female"], patch_artist=True)
    colors = ["lightblue", "pink"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_title("Boxplot över vikt per kön")
    ax.set_xlabel("Kön")
    ax.set_ylabel("Vikt")
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()
def bar_smokers(df):
    counts = df["smoker"].value_counts()
    categories = counts.index
    values = counts.values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories,values, color=["black", "darkgreen"])
    ax.set_title("Andel rökare")
    ax.set_xlabel("Rökare")
    ax.set_ylabel("Antal personer")
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()
