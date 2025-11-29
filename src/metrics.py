import pandas as pd
import numpy as np
import metrics as M
from math import sqrt
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
from scipy.stats import ttest_ind
import viz as v
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("../data/health_study_dataset.csv")


def statistik(df):
    """
    Tar fram statistik för olika kategorier och returnerar en tabel med "mean", "median", "min", "max"

    """


    statistics = df[["age", "weight", "height", "systolic_bp", "cholesterol"]].agg(["mean", "median", "min", "max"])
    return statistics
def sick_ones(df):
    """
    Antal sjuka 
    
    """
    sick = df["disease"] == 1
    counts = sick.sum()
    return counts
def disease_sim(df):

    """
    Simulerar andel sjuka


    Returnar skillanden mellan simulationen och verkilgheten

    """
    counts = M.sick_ones(df)
    people_quantity = len(df)
    disease_count = counts / people_quantity
    np.random.seed(42)
    sim_dis = np.random.choice([0, 1], size=1000, p=[1 - disease_count, disease_count])
    simulated_disease = sim_dis.mean()
    return simulated_disease, sim_dis, disease_count,people_quantity
def true_mean(df):
    """
    Det sanna medelvärdet för systolisk blodtryck

    """

    np.random.seed(123)
    systolic_bp = df["systolic_bp"]
    true_mean = float(np.mean(systolic_bp))
    return true_mean


def ci_systolic_bp(df):
    """
    95% Konfidensintervall med normalappromaxition
    
    """

    data = df["systolic_bp"].dropna()
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)

    z = 1.96
    
    half_width = z * std / sqrt(n)

   
    lo, hi = mean - half_width, mean + half_width
    mean_normal = (lo + hi) / 2
    return lo, hi, mean_normal
def ci_systolic_bp_bootstrap(df, B=5000, confidence=0.95):
    """
    95% Konfidensintervall med bootstrap
    
    """

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
    """
    Hypotespröving - "Rökare har högre blodtryck än ickerökare"

    
    Returnerar testest power
    """
    rokare = df[df["smoker"] == "Yes"] ["systolic_bp"]
    ickerokare = df[df["smoker"] == "No"] ["systolic_bp"]

    t_stat, p_value = stats.ttest_ind(rokare, ickerokare, equal_var=True)
    t_stat_w, p_value_w = stats.ttest_ind(rokare, ickerokare, equal_var=False)


    return rokare, ickerokare, t_stat, p_value, t_stat_w, p_value_w

def power_simulation(rokare, ickerokare, power_simulations=1000, alpha=0.05):
    """
    Power simulation för att se hur säker min hypotesprövning 
    

   
   
    """
    np.random.seed(42)

    significant_count = 0


    n_rok = len(rokare)
    n_icke = len(ickerokare)

    mean_rok = rokare.mean()
    mean_icke = ickerokare.mean()
    std_rok = rokare.std(ddof=1)
    std_icke = ickerokare.std(ddof=1)

    significant_count = 0

    for _ in range(power_simulations):
        sim_rok = np.random.normal(loc=mean_rok, scale=std_rok, size=n_rok)
        sim_icke = np.random.normal(loc=mean_icke, scale=std_icke, size=n_icke)
        t_stat, p_val = stats.ttest_ind(sim_rok, sim_icke, equal_var=False)
        if p_val < 0.05:
            significant_count += 1

    power = significant_count / power_simulations
    return power
class HealthAnalyzer:
    """ 
HealtAnalyzer klass som utför funktioner såsom: 

- Statistik för age, weight, height,systolic_bp, cholesterol från min dataset. 

- En histogram över blodtryck

- En boxplot över vikt grupperat per kön

- En bar som visar andel rökare


    """ 

    def __init__(self, df):
        """ 
        Gör om min df till self 
        
        """
        self.df = df
    def statistik(self):
      
        statistics =  stats = self.df[["age", "weight", "height", "systolic_bp", "cholesterol"]].agg(
            ["mean", "median", "min", "max"]
        )

        return statistics
    
    def hist_blodtryck(self):
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
        
    def box_sex_weight(self):
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

    def bar_smokers(self):
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

def linreg(df):
    """
   En linjär regression för att förutsäga systolisk blodtryck utifrån ålder och vikt.

   Extaraherat variablerna X och y 

   Angett min linreg metod 

   Hämtat koefficienter för ålder, vikt och intercept

   Beräknat r2 

   Gjort sex st prognoser för olika individer. Tre st där ålder är en konstant och tre st där vikten är en konstant.

   Retrun - koefficienter, intercept, r2 och mina 6 prognoser

    """

    X = df[["age", "weight"]].values
    y = df[["systolic_bp"]].values

    linreg = LinearRegression()
    linreg.fit(X, y.ravel())

    coef_age = float(linreg.coef_[0])
    coef_weight = float(linreg.coef_[1])
    intercept = float(linreg.intercept_)

    r2 = float(linreg.score(X, y))

    prediction = float(linreg.predict(np.array([[50, 80]]))[0])
    prediction1 = float(linreg.predict(np.array([[20, 80]]))[0])
    prediction2 = float(linreg.predict(np.array([[15, 80]]))[0])
    prediction3 = float(linreg.predict(np.array([[25, 40]]))[0])
    prediction4 = float(linreg.predict(np.array([[25, 70]]))[0])
    prediction5 = float(linreg.predict(np.array([[25, 100]]))[0])
    

    return coef_age, coef_weight, intercept, r2, prediction, prediction1, prediction2, prediction3, prediction4, prediction5

def run_pca(df):
    """
    Kört PCA på kolumnerna age och weight

    Extraherat variablerna X = ["age", "weight"]

    Standaliserat X med StandardScaler

    Utfört PCA med 2 components

    Return - modellerna och tranformerade 

    """
    X = df[["age", "weight"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2) 
    X_pca = pca.fit_transform(X_scaled)

    return scaler, X_scaled, pca ,X_pca, X

def disease_tabel(df):
    """
    En tabel som visar antal sjuka och procent. 

    percent - gör om till procent och avrundar till en decimal

    disease - antal sjuka

    total - antal personer

    mean_age - medålder per kön

    Returnar en tabell



    """
    total = df.groupby("sex")["disease"].count()

    disease = df[df["disease"] == 1].groupby("sex")["disease"].count()

    percent = (disease / total * 100).round(1)

    mean_age = df[df["disease"] == 1].groupby("sex")["age"].mean().round(1)



    table = pd.DataFrame({
        "Antal sjuka": disease,
        "Totalt": total,
        "Procent sjuka (%)": percent,
        "Medelålder": mean_age
    })
    return table

class advanced_class:
    """
    En klass som hanterar föregående delar av min analys
    """
    def __init__(self, df):
        """ 
        Gör om min df till self 
        
        """
    def disease_tabel(self):
       
        total = df.groupby("sex")["disease"].count()

        disease = df[df["disease"] == 1].groupby("sex")["disease"].count()

        percent = (disease / total * 100).round(1)

        mean_age = df[df["disease"] == 1].groupby("sex")["age"].mean().round(1)
        
        table = pd.DataFrame({
        "Antal sjuka": disease,
        "Totalt": total,
        "Procent sjuka (%)": percent,
        "Medelålder": mean_age
    })

        return table
    def disease_gender(self):
        male_count = df[(df["sex"] == "M") & (df["disease"] == 1)].shape[0]
        female_count = df[(df["sex"] == "F") & (df["disease"] == 1)].shape[0]
    
        labels = ["Male", "Female"]
        counts = [male_count, female_count]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, counts, color= ["blue", "purple"])

        ax.set_title("Sjukdomsförekomst per kön")
        ax.set_xlabel("Kön")
        ax.set_ylabel("Antal med sjukdom")
        ax.grid(axis="y", linestyle= "--", alpha= 0.5)
        plt.tight_layout()
        plt.show()

    def scatter_blodtryck_age(self):
        plt.scatter(df["age"], df["systolic_bp"])
        plt.xlabel("Ålder")
        plt.ylabel("Systolisk blodtryck")
        plt.title("Scatter - Systolisk Blodtryck vs ålder")
        plt.grid(True, linestyle="--", alpha = 0.5)
        plt.tight_layout()
        plt.show()  
    

    def linreg(self):
        X = df[["age", "weight"]].values
        y = df[["systolic_bp"]].values

        linreg = LinearRegression()
        linreg.fit(X, y.ravel())

        coef_age = float(linreg.coef_[0])
        coef_weight = float(linreg.coef_[1])
        intercept = float(linreg.intercept_)

        r2 = float(linreg.score(X, y))

        prediction = float(linreg.predict(np.array([[50, 80]]))[0])
        prediction1 = float(linreg.predict(np.array([[20, 80]]))[0])
        prediction2 = float(linreg.predict(np.array([[15, 80]]))[0])
        prediction3 = float(linreg.predict(np.array([[25, 40]]))[0])
        prediction4 = float(linreg.predict(np.array([[25, 70]]))[0])
        prediction5 = float(linreg.predict(np.array([[25, 100]]))[0])

        
                        
        predictions= pd.DataFrame({
        "Prognos för en person som är 50 år och 80kg ":      [prediction],
        "Prognos för en person som är 20 år och 80kg ":      [prediction1],
        "Prognos för en person som är 15 år och 80kg":       [prediction2],
        "Prognos för en person som är 25 år och 40kg":       [prediction3],
        "Prognos för en person som är 25 år och 70kg":       [prediction4],
        "Prognos för en person som är 25 år och 100kg":      [prediction5]


    })
                       
    
        return predictions