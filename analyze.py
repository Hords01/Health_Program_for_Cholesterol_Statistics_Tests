import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/cholesterol_dataset_labeled.csv', delimiter=',')

# Question 1 -> Test whether the health program changes cholesterol levels.
sns.boxplot(data=[df['cho_init'], df['cho_3mo'], df['cho_6mo']])
plt.xticks([0, 1, 2], ['Before', '3rd Month', '6th Month'])
plt.title('Cholesterol Levels Over Time')
plt.show()

# Question 2 -> Test whether there is a significant difference in glucose levels based on gender.
sns.boxplot(x='gender', y='glucose', data=df)
plt.xticks([0,1], ['Female', 'Male'])
plt.title('Glucose Levels by Gender')
plt.show()

# Question 3 -> Test whether the health program changes the patients weights.
sns.histplot(df['kg_before'], color='blue', label='Before', kde=True)
sns.histplot(df['kg_after'], color='red', label='After', kde=True)
plt.legend()
plt.title('Weight Distribution Before and After the Program')
plt.show()

# Question 4 -> Test whether there is a significant difference in glucose levels among patients across age categories.
sns.boxplot(x='cat_age', y='glucose', data=df)
plt.xticks([0,1,2,3], ['<=40', '41-50', '51-70', '>70'])
plt.title('Glucose Levels by Age Category')
plt.show()

# Question 5 -> Test whether there is a significant difference in the average ages between female and male patients.
sns.barplot(x='gender', y='age', data=df)
plt.xticks([0,1], ['Female', 'Male'])
plt.title('Average Age by Gender')
plt.show()

# Question 6 -> Test whether the health program has changed women's cholesterol levels.
female_df = df[df['gender'] == 1]
sns.boxplot(data=[female_df['cho_init'], female_df['cho_6mo']])
plt.xticks([0, 1], ['Before', 'After'])
plt.title("Women's Cholesterol Levels Before and After the Program")
plt.show()

# Question 7 -> Test whether the health program has changed men's cholesterol levels.
male_df = df[df['gender'] == 2]
sns.boxplot(data=[male_df['cho_init'], male_df['cho_6mo']])
plt.xticks([0, 1], ['Before', 'After'])
plt.title("Men's Cholesterol Levels Before and After the Program")
plt.show()

# Question 8 -> Test whether the health program has changed female patients weights.
sns.histplot(female_df['kg_before'], color='blue', label='Before', kde=True)
sns.histplot(female_df['kg_after'], color='red', label='After', kde=True)
plt.legend()
plt.title("Women's weight Before and After the Program")
plt.show()

# Question 9 -> Test whether the health program has changed male patients weights.
sns.histplot(male_df['kg_before'], color='blue', label='Before', kde=True)
sns.histplot(male_df['kg_after'], color='red', label='After', kde=True)
plt.legend()
plt.title("Men's Weight Before and After the Program")
plt.show()

# Master Graph
# Cholesterol changes (initial, 3 Months, 6 Months)
# Glucose differences by gender
# Weight changes (Before & After)
# Glucose levels across age categories
# Cholesterol changes for Women
# Cholesterol changes for Men

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

df_long = df.melt(value_vars=['cho_init', 'cho_3mo', 'cho_6mo'],
                  var_name='Time', value_name='Cholesterol')
sns.boxplot(x='Time', y='Cholesterol', data=df_long, ax=axes[0, 0])
axes[0, 0].set_title('Cholesterol Levels Over Time')

sns.boxplot(x='gender', y='glucose', data=df, ax=axes[0, 1])
axes[0, 1].set_xticklabels(['Female', 'Male'])
axes[0, 1].set_title('Glucose Levels by Gender')

sns.histplot(df['kg_before'], color='blue', label='Before', kde=True, ax=axes[0, 2])
sns.histplot(df['kg_after'], color='red', label='After', kde=True, ax=axes[0, 2])
axes[0, 2].legend()
axes[0, 2].set_title('Weight Before and After the Program')

sns.boxplot(x='cat_age', y='glucose', data=df, ax=axes[1, 0])
axes[1, 0].set_xticklabels(['<=40', '41-50', '51-70', '>=70'])
axes[1, 0].set_title('Glucose Levels by Age Category')

sns.boxplot(data=[df[df['gender'] == 1]['cho_init'], df[df['gender'] == 1]['cho_6mo']], ax=axes[1, 1])
axes[1, 1].set_xticklabels(['Before', 'After'])
axes[1, 1].set_title("Women's Cholesterol Levels Before and After")

sns.boxplot(data=[df[df['gender'] == 2]['cho_init'], df[df['gender'] == 2]['cho_6mo']], ax=axes[1, 2])
axes[1, 2].set_xticklabels(['Before', 'After'])
axes[1, 2].set_title("Men's Cholesterol Levels Before and After")

plt.tight_layout()
plt.show()