import pandas as pd
import scipy.stats as st
from statsmodels.stats.diagnostic import lilliefors

df = pd.read_csv('data/cholesterol_dataset_labeled.csv', delimiter=',')

# Question 1 -> Test whether the health program changes cholesterol levels.
print('    Question 1 \n -> Test whether the health program changes cholesterol levels.')

print(f"cholesterol initial: {st.shapiro(df.cho_init)}")
print(f"cholesterol 3 months: {st.shapiro(df.cho_3mo)}")
print(f"cholesterol 6 months: {st.shapiro(df.cho_6mo)}")

samples = [df.cho_init, df.cho_3mo, df.cho_6mo]
print(f"Friedman test{st.friedmanchisquare(*samples)}\n")

print("Since p=0.0 < 0.05, it can be said with 95% confidence that the health program has changed the cholesterol levels.\n")

# Question 2 -> Test whether there is a significant difference in glucose levels based on gender.
print("    Question 2 \n -> Test whether there is a significant difference in glucose levels based on gender.")

print(f"Shapiro for gender 1: {st.shapiro(df.glucose[df.gender==1])}")
print(f"Shapiro for gender 2: {st.shapiro(df.glucose[df.gender==2])}")

print(f"Levene test: {st.levene(df.glucose[df.gender==1], df.glucose[df.gender==2])}")
print(f"t-test: {st.ttest_ind(df.glucose[df.gender==1], df.glucose[df.gender==2], equal_var=True)}\n")

print("Since p=0.93 > 0.05, there is no significant difference in glucose levels based on gender at the 5% significance level.\n")

print(f"Test whether women's glucose levels are higher than men's.")
print(f"{st.ttest_ind(df.glucose[df.gender==1], df.glucose[df.gender==2], equal_var=True, alternative='greater')}")

print(f"Test whether women's glucose levels are lower than men's.")
print(f"{st.ttest_ind(df.glucose[df.gender==1], df.glucose[df.gender==2], equal_var=True, alternative='less')}\n")

# Question 3 -> Test whether the health program changes the patients' weights.
print("    Question 3 \n -> Test whether the health program changes the patients' weights.")

print(f"lilliefors test: {lilliefors(df.kg_before - df.kg_after)}")
print(f"shapiro test: {st.shapiro(df.kg_before - df.kg_after)}")
print(f"wilcoxon test: {st.wilcoxon(df.kg_before, df.kg_after)}")
print(f"wilcoxon test -: {st.wilcoxon(df.kg_before - df.kg_after)}")

print("Since p≈0.0 < 0.05, it can be said with 5% significance that the health program has changed the participants' weights.\n")

# Question 4 -> Test whether there is a significant difference in glucose levels among patients across age categories.
print("    Question 4 \n -> Test whether there is a significant difference in glucose levels among patients across age categories.")

print(f"Shapiro cat_age 1 : {st.shapiro(df.glucose[df.cat_age==1])}")
print(f"shapiro cat_age 2 : {st.shapiro(df.glucose[df.cat_age==2])}")
print(f"shapiro cat_age 3 : {st.shapiro(df.glucose[df.cat_age==3])}")
print(f"shapiro cat_age 4 : {st.shapiro(df.glucose[df.cat_age==4])}")

samples_cat_ages = [df.glucose[df.cat_age==1], df.glucose[df.cat_age==2], df.glucose[df.cat_age==3], df.glucose[df.cat_age==4]]
print(f"f_oneway test: {st.f_oneway(*samples_cat_ages)}")

print("Since p=0.63 > 0.05, the null hypothesis (H₀) cannot be rejected. Therefore, at the 95% confidence level, it can be concluded that there is no significant difference in glucose levels among patients across age categories.\n")

# Question 5 -> Test whether there is a significant difference in the average ages between female and male patients.
print("    Question 5 \n -> Test whether there is a significant difference in the average ages between female and male patients.")

print(f"shapiro gender 1: {st.shapiro(df.age[df.gender==1])}")
print(f"shapiro gender 2: {st.shapiro(df.age[df.gender==2])}")

print(f"mannwhitney-u test: {st.mannwhitneyu(df.age[df.gender==1], df.age[df.gender==2])}")

print("Since p=0.20 > 0.05, the null hypothesis (H₀) cannot be rejected. Therefore, at the 5% significance level, it can be concluded that there is no significant difference in the average ages based on gender.\n")

# Question 6 -> Test whether the health program has changed women's cholesterol levels.
print("    Question 6 \n -> Test whether the health program has changed women's cholesterol levels.")

print(f"shapiro cholesterol initial: {st.shapiro(df.cho_init[df.gender==1])}")
print(f"shapiro cholesterol 3 months: {st.shapiro(df.cho_3mo[df.gender==1])}")
print(f"shapiro cholesterol 6 months: {st.shapiro(df.cho_6mo[df.gender==1])}")
samples_cho_1 = [df.cho_init[df.gender==1], df.cho_3mo[df.gender==1], df.cho_6mo[df.gender==1]]
print(f"friedmannchisquare test: {st.friedmanchisquare(*samples_cho_1)}")

print("Since Since p~=0.00 < 0.05, H0 is rejected. That is, at a 5% significance level, there is a significant difference in women's cholesterol levels.\n")

# Question 7 -> Test whether the health program has changed men's cholesterol levels.
print("    Question 7 \n -> Test whether the health program has changed men's cholesterol levels.")

print(f"shapiro cholesterol initial: {st.shapiro(df.cho_init[df.gender==2])}")
print(f"shapiro cholesterol 3 months: {st.shapiro(df.cho_3mo[df.gender==2])}")
print(f"shapiro cholesterol 6 months: {st.shapiro(df.cho_6mo[df.gender==2])}")
samples_cho_2 = [df.cho_init[df.gender==2], df.cho_3mo[df.gender==2], df.cho_6mo[df.gender==2]]
print(f"friedmannchisquare test: {st.friedmanchisquare(*samples_cho_2)}")

print("Since p~=0.00 < 0.05, H0 is rejected. That is, at a 5% significance level, there is a significant difference in men's cholesterol levels.\n")

# Question 8 -> Test whether the health program has changed female patients weights.
print("    Question 8 \n -> Test whether the health program has changed female patients weights.")

print(f"shapiro weights: {st.shapiro(df.kg_before[df.gender==1] - df.kg_after[df.gender==1])}")
print(f"wilcoxon test: {st.wilcoxon(df.kg_before[df.gender==1], df.kg_after[df.gender==1])}")

print("Since p~=0.00, it has been observed that the health program has changed women's weights at a 5% significance level.\n")

# Question 9 -> Test whether the health program has changed male patients weights.
print("    Question 9 -> Test whether the health program has changed male patients' weights.")

print(f"shapiro weights: {st.shapiro(df.kg_before[df.gender==2] - df.kg_after[df.gender==2])}")
print(f"wilcoxon test: {st.wilcoxon(df.kg_before[df.gender==2] - df.kg_after[df.gender==2])}")

print("Since p~=0.00, it can be said that the health program has changed men's weights at a 95% confidence level.")
