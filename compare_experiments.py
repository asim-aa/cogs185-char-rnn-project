import json
import math
import matplotlib.pyplot as plt

experiments = {
    "Baseline (h=128, layers=1)":  "experiments/baseline/loss_history.json",
    "Bigger (h=256, layers=1)":    "experiments/bigger/loss_history.json",
    "Biggest (h=512, layers=1)":   "experiments/biggest/loss_history.json",
    "Deeper (h=128, layers=2)":    "experiments/deeper/loss_history.json",
    "deeper_512 (h=512, layers=2)":"experiments/deeper_512/loss_history.json",
    "sgd_512 (h=512, layers=2, SGD)":"experiments/sgd_512/loss_history.json",
}

plt.figure(figsize=(10, 5))

for label, path in experiments.items():
    with open(path) as f:
        losses = json.load(f)

    final_loss = losses[-1]
    perplexity = math.exp(final_loss)
    print(f"{label}")
    print(f"  Final Loss: {final_loss:.4f}  |  Perplexity: {perplexity:.2f}\n")

    plt.plot(losses, label=label, linewidth=1.8)

plt.title("Training Loss: Architecture Comparison", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/comparison_loss_curve.png", dpi=150)
print("Saved: experiments/comparison_loss_curve.png")