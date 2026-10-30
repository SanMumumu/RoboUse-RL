# RoboDojo tool-using agent integration

Use the [native worker/evaluation guide](../../docs/evaluation.md). The initial integration uses `environments/robodojo/` with separate simulator and transport Python interpreters, original tools, native grading and original cameras.

This directory deliberately contains no simulator assets, recordings, private endpoint/account config or imitation action-head policy. Supply your external compatible RoboDojo/XPolicyLab checkout in a private worker config. A800 does model work; an explicitly allocated RTX renderer does Isaac work. Real end-to-end acceptance remains NOT_RUN.
