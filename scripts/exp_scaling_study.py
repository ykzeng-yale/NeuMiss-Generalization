"""Scaling study: How NeuMiss+MLP performs with more data and higher dimensions."""
import sys; sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
import numpy as np, pandas as pd
from data_generation import DataScenario
from neumiss_plus import NeuMissPlus, NeuMissMLPEstimator, ImputeMLP

# Fixed scenario, vary n_train
def run_data_scaling():
    print("="*60)
    print("DATA SCALING: Gaussian + Quadratic + MCAR, d=10")
    print("="*60)
    results = []
    for n_train in [2000, 4000, 8000, 16000]:
        s = DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10)
        data = s.generate(n_train, n_train//4, 3000, random_state=42)

        configs = [
            ('NeuMiss_d5', lambda: NeuMissPlus(variant='original', depth=5, n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)),
            ('NM+C_d5', lambda: NeuMissPlus(variant='C', depth=5, activation='gelu', expansion_factor=3, n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)),
            ('NM+MLP_d3', lambda: NeuMissMLPEstimator(variant='mlp', depth=3, mlp_layers=(64, 32), n_epochs=150)),
            ('NMwide+MLP_d3', lambda: NeuMissMLPEstimator(variant='wide_mlp', depth=3, expansion_factor=3, mlp_layers=(64, 32), n_epochs=150)),
            ('ImputeMLP', lambda: ImputeMLP(hidden_layers=(128, 64, 32), n_epochs=150)),
        ]
        for name, make_est in configs:
            est = make_est()
            est.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
            pred = est.predict(data['X_test'])
            r2 = 1 - np.mean((data['y_test'] - pred)**2) / np.var(data['y_test'])
            results.append({'n_train': n_train, 'method': name, 'r2_test': r2})
            print(f"  n={n_train:5d} | {name:25s} R2={r2:.4f}")

    return pd.DataFrame(results)

# Fixed n_train, vary dimension
def run_dim_scaling():
    print("\n" + "="*60)
    print("DIMENSION SCALING: Gaussian + Quadratic + MCAR, n=8000")
    print("="*60)
    results = []
    for d in [5, 10, 20, 50]:
        s = DataScenario('gaussian', 'quadratic', 'MCAR', d, 0.5, 10)
        data = s.generate(8000, 2000, 3000, random_state=42)

        configs = [
            ('NeuMiss_d5', lambda: NeuMissPlus(variant='original', depth=5, n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)),
            ('NM+C_d5', lambda: NeuMissPlus(variant='C', depth=5, activation='gelu', expansion_factor=3, n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)),
            ('NM+MLP_d3', lambda: NeuMissMLPEstimator(variant='mlp', depth=3, mlp_layers=(128, 64), n_epochs=150)),
            ('NMwide+MLP_d3', lambda: NeuMissMLPEstimator(variant='wide_mlp', depth=3, expansion_factor=3, mlp_layers=(128, 64), n_epochs=150)),
            ('ImputeMLP', lambda: ImputeMLP(hidden_layers=(128, 64, 32), n_epochs=150)),
        ]
        for name, make_est in configs:
            est = make_est()
            est.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
            pred = est.predict(data['X_test'])
            r2 = 1 - np.mean((data['y_test'] - pred)**2) / np.var(data['y_test'])
            results.append({'n_features': d, 'method': name, 'r2_test': r2})
            print(f"  d={d:3d} | {name:25s} R2={r2:.4f}")

    return pd.DataFrame(results)

# Missing rate scaling
def run_missrate_scaling():
    print("\n" + "="*60)
    print("MISSING RATE SCALING: Gaussian + Quadratic + MCAR, d=10")
    print("="*60)
    results = []
    for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s = DataScenario('gaussian', 'quadratic', 'MCAR', 10, prob, 10)
        data = s.generate(8000, 2000, 3000, random_state=42)

        configs = [
            ('NeuMiss_d5', lambda: NeuMissPlus(variant='original', depth=5, n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)),
            ('NM+MLP_d3', lambda: NeuMissMLPEstimator(variant='mlp', depth=3, mlp_layers=(64, 32), n_epochs=150)),
            ('NMwide+MLP_d3', lambda: NeuMissMLPEstimator(variant='wide_mlp', depth=3, expansion_factor=3, mlp_layers=(64, 32), n_epochs=150)),
            ('ImputeMLP', lambda: ImputeMLP(hidden_layers=(128, 64, 32), n_epochs=150)),
        ]
        for name, make_est in configs:
            est = make_est()
            est.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
            pred = est.predict(data['X_test'])
            r2 = 1 - np.mean((data['y_test'] - pred)**2) / np.var(data['y_test'])
            results.append({'missing_prob': prob, 'method': name, 'r2_test': r2})
            print(f"  p={prob:.1f} | {name:25s} R2={r2:.4f}")

    return pd.DataFrame(results)

df1 = run_data_scaling()
df2 = run_dim_scaling()
df3 = run_missrate_scaling()

# Save all
df1.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_data_scaling.csv', index=False)
df2.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_dim_scaling.csv', index=False)
df3.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_missrate_scaling.csv', index=False)

print("\n\nAll scaling experiments complete!")
