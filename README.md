# FORTRESS-FL: Federated Operator Resilient Trustworthy Resource Efficient Secure Slice Learning

![Python](https://img.shields.io/badge/python-v3.13+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-yellow.svg)

FORTRESS-FL is a Byzantine-robust federated learning system designed for multi-operator network environments. It provides cryptographic commitments, spectral clustering for Byzantine detection, reputation-based aggregation, and differential privacy.

## 🌟 Key Features

- **🛡️ Byzantine Robustness**: Spectral clustering-based Byzantine operator detection
- **🔒 Cryptographic Security**: Pedersen commitments prevent adaptive attacks
- **⚖️ Reputation System**: Dynamic reputation scoring with exponential moving average
- **🔐 Differential Privacy**: Gaussian mechanism for privacy-preserving aggregation
- **🤝 Multi-Party Computation**: Secure cross-operator optimization
- **📊 Comprehensive Evaluation**: Built-in metrics and visualization tools

## 🏗️ System Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Operator 1 │    │  Operator 2 │    │  Operator N │
│   θ₁, 𝒟₁    │    │   θ₂, 𝒟₂    │    │   θₙ, 𝒟ₙ    │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       │ Commit(g₁)       │ Commit(g₂)       │ Commit(gₙ)
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
              ┌───────────▼───────────┐
              │     Coordinator       │
              │  • Verify commits     │
              │  • Spectral cluster   │
              │  • Update reputation  │
              │  • Aggregate θ*       │
              └───────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/fortress-fl.git
cd fortress-fl

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from fortress_fl import FortressFL, train_fortress_fl, create_federated_datasets

# Create federated datasets
operators_data = create_federated_datasets(
    n_operators=5,
    n_samples_per_operator=100,
    n_features=10,
    byzantine_operators=[3, 4],  # Last 2 are Byzantine
    task_type='regression'
)

# Train FORTRESS-FL
final_model, history = train_fortress_fl(
    operators_data=operators_data,
    n_rounds=20,
    model_dim=10,
    learning_rate=0.01,
    verbose=True
)

print(f"Training completed! Final model norm: {np.linalg.norm(final_model):.4f}")
```

### Advanced Usage

```python
# Initialize FORTRESS-FL with custom parameters
fortress_fl = FortressFL(
    n_operators=8,
    model_dim=15,
    security_param=2048,    # Cryptographic security
    lambda_rep=0.15,        # Reputation update rate
    sigma_dp=0.08,          # DP noise level
    epsilon_dp=0.12         # Privacy budget per round
)

# Execute training rounds
for round_idx in range(10):
    local_gradients = get_operator_gradients()  # Your gradient computation
    result = fortress_fl.train_round(local_gradients, learning_rate=0.01)

    print(f"Round {round_idx + 1}: {result['n_honest']} honest, "
          f"{result['n_byzantine']} Byzantine detected")
```

## 📚 Examples

### 1. Simple Example

```bash
python examples/simple_example.py
```

### 2. Comprehensive Demo

```bash
python examples/fortress_fl_demo.py
```

This runs a complete demonstration including:
- Multi-operator federated learning
- Byzantine attack scenarios
- Privacy-utility tradeoff analysis
- Performance evaluation
- MPC cross-operator optimization

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test modules
python tests/test_crypto.py
python tests/test_aggregation.py
python tests/test_end_to_end.py
```

## 📖 Algorithm Overview

### 1. Cryptographic Commitment Phase

Operators commit to gradients using Pedersen commitments to prevent adaptive attacks:

```
C_i = g^{H(g_i)} · h^{r_i} mod p
```

### 2. Spectral Clustering for Byzantine Detection

Partition operators based on gradient similarity using normalized graph Laplacian:

```
L_norm = I - D^{-1/2} A D^{-1/2}
```

### 3. Reputation-Based Aggregation

Weight gradients by reputation scores with exponential moving average:

```
r_i^{t+1} = (1-λ)r_i^t + λ·Q_i^t
θ* = Σ w_i · g_i where w_i = r_i / Σr_j
```

### 4. Differential Privacy

Add calibrated Gaussian noise for privacy:

```
θ*_DP = θ* + N(0, σ²I)
```

## 🔧 Configuration

Key hyperparameters:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `security_param` | Cryptographic security (bits) | 2048 | 1024-4096 |
| `lambda_rep` | Reputation update rate | 0.1 | 0.01-0.5 |
| `sigma_dp` | DP noise standard deviation | 0.1 | 0.01-1.0 |
| `epsilon_dp` | Privacy budget per round | 0.1 | 0.01-1.0 |

## 🎯 Performance

Typical performance on modern hardware:

- **Setup Time**: ~2-5s (2048-bit security)
- **Round Time**: ~0.5-2s (10 operators, 50D model)
- **Memory Usage**: ~100-500MB (depends on model size)
- **Byzantine Detection**: 80-95% accuracy

## 📊 Evaluation and Visualization

FORTRESS-FL includes comprehensive evaluation tools:

```python
from fortress_fl.utils.evaluation import (
    evaluate_convergence, evaluate_byzantine_robustness,
    plot_training_metrics, generate_performance_report
)

# Evaluate system performance
convergence = evaluate_convergence(fortress_fl)
robustness = evaluate_byzantine_robustness(fortress_fl, byzantine_operators)
report = generate_performance_report(fortress_fl, test_data, byzantine_operators)

# Generate visualizations
plot_training_metrics(fortress_fl, save_path='training_metrics.png')
```

## 🛡️ Security Features

### Threat Model
- Up to f < n/3 Byzantine operators
- Adaptive adversaries with algorithm knowledge
- No trusted central authority

### Defense Mechanisms
1. **Commitment Scheme**: Prevents gradient adaptation attacks
2. **Spectral Clustering**: Detects coordinated Byzantine behavior
3. **Reputation System**: Accumulates trust over time
4. **Differential Privacy**: Protects individual contributions

### Attack Resilience
- ✅ Sign flip attacks
- ✅ Random noise attacks
- ✅ Model poisoning attacks
- ✅ Coordinated attacks
- ✅ Adaptive attacks (partial)

## 🤝 Multi-Party Computation

FORTRESS-FL supports secure cross-operator optimization:

```python
from fortress_fl.crypto.mpc import secure_joint_optimization_mpc

# Secure interference optimization between operators
power_A, power_B = secure_joint_optimization_mpc(
    operator_A_data={'interference_matrix': I_A, 'power_range': (0.1, 2.0)},
    operator_B_data={'interference_matrix': I_B, 'power_range': (0.1, 2.0)},
    n_iterations=15
)
```

## 📁 Project Structure

```
fortress_fl/
├── crypto/              # Cryptographic components
│   ├── pedersen.py     # Pedersen commitments
│   └── mpc.py          # Multi-party computation
├── aggregation/         # Aggregation algorithms
│   ├── spectral_clustering.py
│   ├── reputation.py
│   └── trustchain.py
├── core/                # Main FORTRESS-FL implementation
│   ├── fortress_fl.py  # Core FortressFL class
│   └── training.py     # Training functions
├── utils/               # Utilities and evaluation
│   ├── attacks.py      # Byzantine attack generators
│   └── evaluation.py   # Evaluation and metrics
└── examples/            # Usage examples
    ├── simple_example.py
    └── fortress_fl_demo.py
```

## 🔬 Research Background

FORTRESS-FL implements state-of-the-art techniques from:

- **Byzantine-Robust Aggregation**: Spectral clustering approach
- **Cryptographic Commitments**: Pedersen commitment scheme
- **Reputation Systems**: Exponential moving average updates
- **Differential Privacy**: Gaussian mechanism with composition
- **Federated Learning**: Multi-operator network orchestration

## 📝 Citation

If you use FORTRESS-FL in your research, please cite:

```bibtex
@software{fortress_fl_2024,
  title={FORTRESS-FL: Byzantine-Robust Federated Learning for Network Orchestration},
  author={FORTRESS-FL Team},
  year={2024},
  url={https://github.com/your-org/fortress-fl}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- 📧 Email: fortress-fl-support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/fortress-fl/issues)
- 📖 Documentation: [Wiki](https://github.com/your-org/fortress-fl/wiki)

## 🎉 Acknowledgments

Special thanks to the research community for foundational work in:
- Byzantine-robust federated learning
- Cryptographic protocols for distributed systems
- Privacy-preserving machine learning
- Network orchestration and optimization

---

**Built with ❤️ for secure and robust federated learning**