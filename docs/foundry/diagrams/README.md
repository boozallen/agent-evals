# Architecture Diagrams

High-level diagrams explaining the key architectural innovations in the Agent Evals framework.

## Diagrams

### 1. [System Architecture](system-architecture.md)
Hexagonal architecture showing layer organization and dependency relationships. Core defines port interfaces, adapters implement them, with strict dependency inversion enforced.

### 2. [Evaluation Modes](evaluation-modes.md)
Unified interface for both live evaluation (execute agents) and historical evaluation (score saved traces). Same framework, scorers, and result format for both modes.

### 3. [Adapter Architecture](adapter-architecture.md)
Platform and scorer adapters are fully independent and extensible - any platform works with any scorer combination without vendor lock-in. Third parties can add custom adapters without modifying core code.

### 4. [CI/CD Integration](cicd-regression-testing.md)
Automated quality gates in CI/CD pipelines catch agent regressions before production, preventing "whack-a-mole" problems in development.

### 5. [Evaluation Lifecycle](eval-lifecycle.md)
End-to-end workflow from identifying failure modes through building eval packages, execution, and iterative improvement. Covers both live and historical evaluation modes.
