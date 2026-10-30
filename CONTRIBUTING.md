# Contributing

Keep Policy, Harness, Tools, Controller, Environment and Trainer responsibilities explicit. Prefer extending a tested adapter over creating a second optimization/distributed framework. Run the relevant CPU tests; GPU/native tests must use explicit allocations and budgets. Tests must not silently start CUDA, a simulator or a paid teacher call.

Include the exact configuration, seed, data identity and native evidence for experiment claims. Label synthetic fixtures, failed/unknown results and NOT_RUN acceptance honestly. Do not introduce test leakage, teacher-as-student on-policy data or repaired behavior log probabilities. Never commit credentials, private logs, model weights, raw datasets or recordings.

Contributions should target a branch and Pull Request, with implementation/maturity and actual validation evidence in the PR description. See `docs/architecture.md` for extension contracts and `docs/reproducibility.md` for checks.
