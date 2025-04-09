import pickle

with open("reg_exp_results.pkl", "rb") as f:
    results = pickle.load(f)

results = sorted(results, key=lambda x: x[0])
print(results[0])