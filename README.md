# data_assimilation

This repository studies a data assimilation framework for a phase-field model of thrombus formation and blood flow.

## Model description

The governing system is based on the phase-field model introduced in *Local Well-Posedness of a Three-Dimensional Phase-Field Model for Thrombus and Blood Flow* by Woojeong Kim, Krutika Tawri, and Roger Temam.

In the present work, the original model is modified by adding an additional diffusion term.

The governing equations are:

```math
\rho\left(\frac{\partial u}{\partial t} + u \cdot \nabla u\right)
+ \nabla p
- \nabla \cdot \left(\eta(\phi)\nabla u\right)
=
-\lambda \nabla \cdot (\nabla \phi \otimes \nabla \phi)
+ \nabla \cdot \left(\lambda_e (1-\phi)(F F^T - I)\right)
- \eta(\phi)\frac{(1-\phi)u}{\kappa(\phi)}
```

```math
\nabla \cdot u = 0
```

```math
\frac{\partial F}{\partial t} + u \cdot \nabla F = \nabla u \, F
```

```math
\frac{\partial \phi}{\partial t} + u \cdot \nabla \phi = \tau \Delta \mu
```

```math
\mu = -\lambda \Delta \phi + \lambda \gamma f'(\phi)
- \frac{\lambda_e}{2}\mathrm{tr}(F F^T - I)
```

## Main results

Data assimilation was successfully implemented in the following two settings.

### 1. Case with prescribed initial conditions

In this setting, the initial conditions are given in advance.

- The data assimilation algorithm was successfully implemented with prescribed initial data.
- The assimilated solution showed stable convergence toward the reference solution.
- The numerical results confirmed effective recovery of the target dynamics.
- This case demonstrates that the proposed framework performs reliably when sufficient initial information is available.

### 2. Case without prescribed initial conditions

In this setting, the initial conditions are not given in advance.

- The data assimilation algorithm was also successfully implemented without prescribed initial data.
- Even without knowing the initial state beforehand, the assimilated solution was able to recover the reference dynamics from observational data.
- The method remained stable throughout the assimilation process.
- This case shows that the proposed framework is robust even when the initial state is unknown.

## Summary

The proposed data assimilation framework was successfully validated in both cases:

- with prescribed initial conditions,
- without prescribed initial conditions.

In both settings, the method achieved stable and effective recovery of the target dynamics.
