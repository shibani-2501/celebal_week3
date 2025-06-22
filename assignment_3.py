import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
df = pd.read_csv(url)

print(df.head())
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Fill Age with median, Embarked with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop "Cabin" (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# Bar Graph - Survival count
survived_count = df['Survived'].value_counts()
plt.bar(['Not Survived', 'Survived'], survived_count, color=['red', 'green'])
plt.title('Survival Count')
plt.ylabel('Number of Passengers')
plt.show()

# Bar Graph - Survival by gender
gender_survival = df.groupby('Sex')['Survived'].value_counts().unstack()
gender_survival.plot(kind='bar', stacked=True, color=['red', 'green'])
plt.title('Survival by Gender')
plt.ylabel('Number of Passengers')
plt.show()

plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

df.boxplot(column='Age', by='Pclass', grid=False)
plt.title('Age by Passenger Class')
plt.suptitle('')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

# Scatter plot - Age vs Fare
plt.scatter(df['Age'], df['Fare'], alpha=0.6, color='purple', edgecolor='k')
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True)
plt.show()