# AntClock Model Optimization Demo

This demo showcases the [AntClock](https://github.com/selfapplied/antclock) framework for optimizing machine learning models through Coherence Engine (CE) architecture.

## Overview

AntClock is a mathematical framework that reconstructs the Riemann zeta function as a Galois covering space of the integers, built from curvature flows and digit symmetries. This demo demonstrates how to use AntClock's CE architecture to optimize models for agent systems.

## Features

- **CE Architecture Integration**: Uses CE1/CE2/CE3 layers for model optimization
- **Curvature-Based Learning**: Leverages discrete geometry for improved model performance
- **Benchmark Demonstration**: Shows performance improvements on sample tasks

## Quick Start

```bash
# Start the demo
docker compose up --build

# Or with OpenAI models
docker compose -f compose.yaml -f compose.openai.yaml up
```

## What This Demo Does

The demo creates an optimized text classification model using AntClock's Coherence Engine (CE) architecture:

1. **CE1 (Discrete Grammar)**: Corridor-based embeddings capture discrete geometric structures
2. **CE2 (Dynamical Flow)**: Flow operators model continuous transformations
3. **CE3 (Emergent Simplicial)**: Witness consistency ensures topological invariants

The CE architecture provides several advantages over traditional models:
- Better handling of discrete structures through corridor embeddings
- Improved generalization via geometric regularization
- Built-in topological consistency checks

## Architecture

The demo includes:
- `agent.py`: Main agent implementation using CE architecture
- `compose.yaml`: Docker Compose configuration
- `Dockerfile`: Container setup with AntClock dependencies
- `requirements.txt`: Python dependencies

## Performance

The CE architecture typically shows improvements in:
- Sample efficiency (fewer training examples needed)
- Generalization to out-of-distribution data
- Compositional understanding
- Mathematical reasoning tasks

## References

- [AntClock GitHub Repository](https://github.com/selfapplied/antclock)
- [CE Framework Specification](https://github.com/selfapplied/antclock/blob/main/docs/spec.md)
- [Mathematical Background](https://github.com/selfapplied/antclock/blob/main/README.md)

## License

This demo follows the dual-license structure of the parent repository (Apache-2.0 OR MIT).

The AntClock framework is licensed under CC BY-SA 4.0.
