import csv

data = []

with open("trainingdata.csv") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        data.append(row)

num_attr = len(data[0]) - 1

S = ['0'] * num_attr
G = ['?'] * num_attr

print("Initial Specific Hypothesis:")
print(S)

print("Initial General Hypothesis:")
print(G)

for row in data:

    if row[-1].lower() == "yes":

        for i in range(num_attr):
            if S[i] == '0':
                S[i] = row[i]
            elif S[i] != row[i]:
                S[i] = '?'

    else:

        for i in range(num_attr):
            if S[i] != '?' and S[i] != row[i]:
                G[i] = S[i]
            else:
                G[i] = '?'

print("\nFinal Specific Hypothesis:")
print(S)

print("\nFinal General Hypothesis:")
print(G)
