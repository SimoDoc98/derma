# Theory

DERMA works in two stages. The **estimation** stage measures a set of interpretable
parameters on real, decomposed EDA recordings and summarizes their population
variability as fitted distributions (a **profile**). The **generation** stage samples one
parameter set from a profile and synthesizes the corresponding signal, returning its
tonic and phasic components as ground truth.

## Signal model

The skin conductance is the sum of a slow tonic and a fast phasic component, as in the
continuous deconvolution framework of cvxEDA (Greco et al., 2016):

```math
y(t) = y_{\text{tonic}}(t) + y_{\text{phasic}}(t)
```

The two components are not independent: a shared sudomotor driver sets the onsets of the
phasic events and, through the SCR/SCL ratio, the absolute tonic level (Bach et al., 2011).

## Phasic component

```math
y_{\text{phasic}}(t) = \max\Big(0,\ \left[u \otimes h_{\text{base}}\right](t) + \left[u_{\text{diff}} \otimes h_{\text{deriv}}\right](t)\Big)
```

**Sudomotor driver.** A sparse train of bursts,

```math
u(t) = \sum_{i=1}^{N_e} A_i\,\delta(t - t_i), \qquad N_e = \lfloor \nu\,T/60 \rfloor ,
```

with $\nu$ the NS.SCR frequency (events/min) over a window of $T$ seconds. Onsets are
drawn at random with a refractory period $T_r = 1$ s between bursts; amplitudes are
$A_i \sim \text{Lognormal}(0, \sigma_A)$, non-negative and right-skewed.

**SCR kernel.** The sudomotor response is a third-order, purely damped ODE (Bach et al., 2011):

```math
\dddot{x} + \theta_1\,\ddot{x} + \theta_2\,\dot{x} + \theta_3\,x = u
```

To keep it stable and free of ringing, the system is parameterized by three real
negative poles $-\lambda_1, -\lambda_2, -\lambda_3$ ($\lambda_k > 0$, 1/s), and the
coefficients follow from Vieta's formulas:
$\theta_1 = \lambda_1 + \lambda_2 + \lambda_3$,
$\theta_2 = \lambda_1\lambda_2 + \lambda_1\lambda_3 + \lambda_2\lambda_3$,
$\theta_3 = \lambda_1\lambda_2\lambda_3$.

The poles are tied to an interpretable morphology by calibrating the ODE against the
canonical SCR shape of Bach et al. (2010): a Gaussian sweat diffusion convolved with a
bi-exponential recovery, normalized to unit peak,

```math
h_{\text{target}}(t) \propto \mathcal{N}\!\left(t;\ 2.2\,t_{\text{rise}},\ t_{\text{rise}}\right) \otimes \left(e^{-t/\tau_1} + e^{-t/\tau_2}\right)
```

```math
\boldsymbol{\lambda}^{*} = \arg\min_{\boldsymbol{\lambda} > 0}\ \left\| h_{\text{target}}(t) - h_{\text{ODE}}(t - \Delta;\ \boldsymbol{\lambda}) \right\|_2^2
```

where $\Delta$ aligns the two peaks, so that the fit targets the shape and not the
latency of the causal ODE. The fitted response, scaled to unit peak, is $h_{\text{base}}$:
driver amplitudes are therefore SCR amplitudes in uS.

**Shape variability.** Consecutive SCRs of the same subject differ slightly in shape.
This is modeled by an informed basis set (Bach et al., 2010; 2013): the derivative kernel
$h_{\text{deriv}} = \mathrm{d}h_{\text{base}}/\mathrm{d}t$ is driven by
$u_{\text{diff}}(t) = \sum_i \beta_i A_i\,\delta(t - t_i)$, with
$\beta_i \sim \mathcal{N}(0, \sigma_\beta)$ and $\sigma_\beta = 0.2$.

## Tonic component

```math
y_{\text{tonic}}(t) = m\,t + A_{\text{osc}} \sin\!\left(\frac{2\pi}{T_{\text{osc}}}\,t\right) + L_0
```

The drift slope $m$ is fitted on the data. $A_{\text{osc}}$ and $T_{\text{osc}}$ are drawn
from uniform literature ranges instead: fitting a sinusoid to a drifting tonic over a
finite window is ill-posed and highly sensitive to the window length.

The absolute level is coupled to the phasic activity through the SCR/SCL ratio $\rho$:

```math
L_0 = \min\!\left(25,\ \max\!\left(1,\ \frac{A_{\text{ph}}}{\rho}\right)\right)\ \mu\text{S}
```

with $A_{\text{ph}}$ the mean of the phasic samples above 0.05 uS.

## Parameters

| Symbol | Profile field | Unit | Literature range | Source | Fitted |
|---|---|---|---|---|---|
| $m$ | `scl_drift` | uS/s | [-0.01, 0.01] | Bach et al., 2010 | yes, support $\mathbb{R}$ |
| $\nu$ | `nsscr` | events/min | [1, 25] | Braithwaite et al., 2013 | yes, $\geq 0$ |
| $\sigma_A$ | `scr_amp_sd` | - | [0.1, 0.5] | Braithwaite et al., 2013 | yes, $> 0$ |
| $t_{\text{rise}}$ | `scr_rise_time` | s | [0.5, 2.5] | Leiner et al., 2012 | yes, $> 0$ |
| $\tau_1$ | `scr_tau1` | s | [0.5, 2.0] | Leiner et al., 2012 | yes, $> 0$ |
| $\tau_2/\tau_1$ | `scr_tau_ratio` | - | [2.0, 10.0] | Leiner et al., 2012 | yes, $> 0$ |
| $\rho$ | `scr_scl_ratio` | - | [0.1, 1] | Braithwaite et al., 2013 | yes, $> 0$ |
| $T_{\text{osc}}$ | `scl_period_range` | s | [120, 600] | Greco et al., 2016 | no |
| $A_{\text{osc}}$ | - | uS | [0.01, 0.5] | Greco et al., 2016 | no |
| $T_r$ | - | s | 1 | Greco et al., 2016 | no |
| $\sigma_\beta$ | `scr_beta_var` | - | 0.2 | - | no |

The literature profile (`derma.profiles.literature_profile`) draws every fitted
parameter uniformly within its range.

## Estimation

Recordings are first decomposed into tonic and phasic components (e.g. with
`derma.decomposition.decompose`) and cut into segments of the conditions of interest.
Each segment gives one value per parameter:

- $m$: slope of the least-squares line through the tonic;
- SCRs: peaks of the phasic, onset at the left base of the peak prominence;
- $\nu$: number of SCRs per minute;
- $\sigma_A$: coefficient of variation of the SCR amplitudes (peak minus onset);
- $t_{\text{rise}}$: mean onset-to-peak time;
- $\tau_1$, $\tau_2$: mean time from the peak until the SCR decays to 67% and 36.8% of
  its amplitude;
- $\rho$: mean ratio between phasic and tonic at the SCR peaks.

The measured $t_{\text{rise}}$, $\tau_1$ and $\tau_2$ enter $h_{\text{target}}$ as its
shape parameters.

A single segment can be used directly as a fixed parameter set (single-subject setting).
In the population setting, the segment values of each parameter are fitted by maximum
likelihood to a set of candidate families, and the family with minimum AIC (Akaike, 1974)
is kept, together with the observed range as bounds:

- positive parameters: Lognormal, Gamma, Exponential, Normal, Beta, Gompertz, Johnson $S_B$;
- NS.SCR, which keeps its zeros: Exponential, Half-normal, Gamma, Normal;
- drift: Normal, Student's $t$, Logistic, Laplace.

The generator samples each parameter from its family, rejecting values outside the bounds.

## Goodness of fit

`derma.report` checks each fitted distribution with the Kolmogorov-Smirnov test
(statistic $D$ and p-value) against the segment values. It also fits the ODE kernel to
the median morphology ($t_{\text{rise}}$, $\tau_1$, $\tau_2$) of the profile and reports
the mean squared error against $h_{\text{target}}$, to verify that the ODE constraints
preserve the intended SCR shape.

## References

- Akaike H (1974). A new look at the statistical model identification. *IEEE Trans Autom Control* 19(6):716-723.
- Bach DR, Flandin G, Friston KJ, Dolan RJ (2010). Modelling event-related skin conductance responses. *Int J Psychophysiol* 75(3):349-356.
- Bach DR, Daunizeau J, Kuelzow N, Friston KJ, Dolan RJ (2011). Dynamic causal modeling of spontaneous fluctuations in skin conductance. *Psychophysiology* 48(2):252-257.
- Bach DR, Friston KJ, Dolan RJ (2013). An improved algorithm for model-based analysis of evoked skin conductance responses. *Biol Psychol* 94(3):490-497.
- Braithwaite JJ, Watson DG, Jones R, Rowe M (2013). A guide for analysing electrodermal activity (EDA) & skin conductance responses (SCRs) for psychological experiments. University of Birmingham.
- Greco A, Valenza G, Lanata A, Scilingo EP, Citi L (2016). cvxEDA: a convex optimization approach to electrodermal activity processing. *IEEE Trans Biomed Eng* 63(4):797-804.
- Leiner D, Fahr A, Früh H (2012). EDA positive change: a simple algorithm for electrodermal activity to measure general audience arousal during media exposure. *Commun Methods Meas* 6(4):237-250.
