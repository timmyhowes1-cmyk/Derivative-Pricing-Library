# Derivative Pricing Engine in Python

An object-oriented Python library for pricing financial derivatives and computing Greeks.
## Overview

This project is a personal quantitative finance build focused on designing a reusable derivative pricing framework in Python using object-oriented programming. The codebase is structured so that new instruments, stochastic models, and pricing methods can be added with minimal changes to the core architecture.

The project was built to strengthen practical skills in:
- Derivative pricing
- Numerical methods
- Monte Carlo simulation
- Object-oriented software design

## Features

- Object-oriented architecture for instruments, models, and pricing engines
- Modular design to support extension to new pricing methods
- Monte Carlo pricing functionality
- Greeks calculation and sensitivity analysis
- Numerical scheme implementation for stochastic processes
- Validation against analytical benchmarks where available

## Repository Structure

```text
.
├── instruments/     # Option and derivative product definitions
├── models/          # Model classes (e.g. Black-Scholes, Heston)
├── engines/         # Pricing engines and numerical methods
├── examples/        # Example scripts / notebooks
└── README.md
```

## Architecture

The code follows a modular object-oriented design:

- **Instrument classes** define payoff and contract specifications
- **Model classes** define market dynamics and model parameters
- **Engine classes** implement pricing logic and Greek calculations
- **Numerical Scheme classes** discretisation scheme to use for continuous model

## Implemented Functionality

Current functionality includes:
- European and some exotic equity option pricing
- Monte Carlo path simulation (with antithetic variates option)
- Greek estimation
- Analytical and numerical result comparison for BSM and Heston models
- Extensible pricing engine framework


## Why I Built This

I built this project to deepen my understanding of derivative pricing, quantitative modelling, and Python software design. The goal was not only to price instruments, but to design a codebase that reflects how a reusable quant library might be structured in practice.
