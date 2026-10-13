import csv, math
from collections import Counter
from matplotlib import pyplot as plt

data = []
palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
category_colors = {}
categories = []

with open('iris.csv', mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    rows = list(csv_reader)

    for row in rows:
        data.append({
            f'{header[2]}' : float(row[2]),
            f'{header[3]}' : float(row[3]),
            f'{header[4]}' : row[4]
        })
        cat_value = row[4]
        categories.append(cat_value)
        if cat_value not in category_colors:
            category_colors[cat_value] = palette[len(category_colors) % len(palette)]

x_key = header[2]
y_key = header[3]
cat_key = header[4]

plt.figure(figsize=(8, 6))

for cat_value, color in category_colors.items():
    x_vals = [row[x_key] for row in data if row[cat_key] == cat_value]
    y_vals = [row[y_key] for row in data if row[cat_key] == cat_value]
    
    plt.scatter(
        x_vals, 
        y_vals, 
        color=color, 
        label=cat_value, 
        alpha=0.8, 
        edgecolors='k', 
        s=50
    )

input_val = {
    'k': 18,
    'x' : 4.0,
    'y' : 1.4
}
distances = []
for row in data:
    dist = math.sqrt((row[f'{x_key}'] - input_val['x']) ** 2 + (row[f'{y_key}'] - input_val['y']) ** 2)
    distances.append((dist, row))
distances.sort(key=lambda x : x[0])
k_nearest = distances[:input_val['k']]
neighbor_categories = [neighbor[f'{cat_key}'] for dist, neighbor in k_nearest]
vote_counts = Counter(neighbor_categories)
predicted_category, count = vote_counts.most_common(1)[0]
print(f"Vote breakdown: {vote_counts}")
print(f"Predicted Category for the new point: {predicted_category} ({count}/{input_val['k']} neighbors)")

for dist, neighbor in k_nearest:
    plt.plot(
        [input_val['x'], neighbor[f'{x_key}']],
        [input_val['y'], neighbor[f'{y_key}']],
        color='black',
        linestyle='--',
        alpha = 0.7,
        linewidth = 1
    )
plt.plot(input_val['x'], input_val['y'], 'r*', label=f'New Input, Predicted species:{predicted_category}' )
plt.xlabel(x_key)
plt.ylabel(y_key)
plt.title(f'KNN Prediction where k={input_val['k']}, {x_key} = {input_val['x']}, {y_key} = {input_val['y']}')
plt.legend(title='Species')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
