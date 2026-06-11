t = [1,2,3,4,5,6,7,8,2,3,4,5,6]
temp = [70,69,68,67,66,65,64,63,75,74,73,72,71]

groups = {}
for time, temperature in zip(t, temp):
    groups.setdefault(time, []).append(temperature)

unique_t = sorted(groups.keys())
avg_temp = [sum(groups[time]) / len(groups[time]) for time in unique_t]

print(unique_t)
print(avg_temp)