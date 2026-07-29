import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Training Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
                'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
                'Sunny', 'Overcast', 'Overcast', 'Rain'],

    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],

    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],

    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Strong'],

    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes',
             'No', 'Yes', 'No', 'Yes', 'Yes',
             'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Convert categorical data into numerical values
outlook_map = {'Sunny': 0, 'Overcast': 1, 'Rain': 2}
temp_map = {'Hot': 0, 'Mild': 1, 'Cool': 2}
humidity_map = {'High': 0, 'Normal': 1}
wind_map = {'Weak': 0, 'Strong': 1}
play_map = {'No': 0, 'Yes': 1}

df['Outlook'] = df['Outlook'].map(outlook_map)
df['Temperature'] = df['Temperature'].map(temp_map)
df['Humidity'] = df['Humidity'].map(humidity_map)
df['Wind'] = df['Wind'].map(wind_map)
df['Play'] = df['Play'].map(play_map)

# Split features and target
X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df['Play']

# Train Decision Tree using Entropy (ID3)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# User Input
print("\nEnter New Sample Details")

outlook = input("Outlook (Sunny/Overcast/Rain): ")
temperature = input("Temperature (Hot/Mild/Cool): ")
humidity = input("Humidity (High/Normal): ")
wind = input("Wind (Weak/Strong): ")

sample = [[
    outlook_map[outlook],
    temp_map[temperature],
    humidity_map[humidity],
    wind_map[wind]
]]

prediction = model.predict(sample)

print("\nPrediction:")

if prediction[0] == 1:
    print("Play = Yes")
else:
    print("Play = No")
