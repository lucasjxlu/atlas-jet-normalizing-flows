# Conditional normalizing flows for ATLAS jet simulation

This repository accompanies my undergraduate physics thesis, **“Normalizing Flows for Detector-Level Jet Generation in Fully Hadronic $t\bar{t}$ Events.”** It investigates whether a conditional normalizing flow can provide a fast, data-driven map from truth-level jets to reconstructed ATLAS jet features while retaining the correlations needed for physics analysis.

The [full thesis](docs/Jiaxun_Lu_Physics_Thesis_2026.pdf) gives the complete motivation, derivations, and discussion. This README is a shorter guide to the physics question, method, and principal results.

## Physics problem and background

Large Hadron Collider analyses depend on simulated collision events. A conventional production chain generates the hard collision, propagates particles through a detailed detector model, digitizes detector signals, and reconstructs analysis-level objects. This is accurate but computationally expensive, motivating generative models that can learn parts of the map from truth-level particles to reconstructed detector objects.

This project studies fully hadronic top-quark pair production:

$$
t\bar{t}\rightarrow (W^+b)(W^-\bar{b})
\rightarrow (q\bar{q}'b)(q\bar{q}'\bar{b}).
$$

After hadronization, the six primary quarks are observed as jets: two bottom-flavoured jets and four light-flavour jets from the two $W$ bosons. Each reconstructed jet is described by its transverse momentum $p_T$, pseudorapidity $\eta$, azimuthal angle $\phi$, mass $m$, and four GN2 flavour-tagging probabilities.

This final state is a useful stress test for fast simulation. Matching each jet feature separately is not enough: reconstructing the two $W$ bosons and top quarks depends on flavour identification, jet pairing, four-momentum conservation, and correlations across the whole event. The reconstructed top-mass distribution is therefore used here as a downstream test of physical fidelity.

## Methodology

### Data and event representation

The analysis uses the public [ATLAS $t\bar{t}$ JetSet dataset](https://opendata.cern.ch/record/atlas-93940), containing simulated proton-proton collisions at $\sqrt{s}=13.6$ TeV. The preprocessing pipeline:

1. converts momenta and masses from MeV to GeV;
2. applies $p_T>30$ GeV and $|\eta|<2.5$ selections;
3. retains events with exactly six selected reconstructed jets;
4. sorts jets by decreasing $p_T$;
5. transforms the four constrained flavour probabilities into
   $z_b=\log(p_b/p_u)$, $z_c=\log(p_c/p_u)$, and
   $z_\tau=\log(p_\tau/p_u)$; and
6. performs an 80/20 train-test split at event level.

The resulting target is a 42-dimensional reconstructed event vector: six jets with seven features each. The model is conditioned on a 24-dimensional vector containing four truth-level features for each jet.

### Conditional normalizing flow

A normalizing flow learns an invertible transformation between a simple latent density and the detector-level target distribution. Because the transformation is invertible, the exact likelihood can be evaluated using the change-of-variables formula. Conditioning every transformation on truth-level jet information makes the learned distribution event-dependent:

$$
p(\text{reconstructed jets}\mid\text{truth jets}).
$$

The implementation uses autoregressive rational-quadratic spline transformations with permutation layers between blocks. The production configuration contains 12 flow blocks, three hidden layers of 128 units, 16 spline bins, and a 42-dimensional diagonal-Gaussian base distribution. It is trained for 100 epochs with Adam, a learning rate of $10^{-6}$, batch size 1024, and seed 121.

### Physics validation

Generated events are evaluated at three increasingly demanding levels:

- one-dimensional kinematic and flavour-score distributions;
- Pearson correlations among reconstructed features; and
- reconstructed $W$-boson and top-quark masses.

For mass reconstruction, the two jets with the strongest bottom-jet discriminants are selected as $b$ candidates. The remaining four jets are paired into two $W$ candidates by minimizing their disagreement with $m_W\approx80.4$ GeV. The two possible $b$-to-$W$ assignments are then compared, and the assignment giving the most mutually consistent top masses is retained.

## Results

The flow reproduces the main reconstructed kinematic distributions, including their peaks and tails. The flavour-tagging probabilities are more difficult because several are sharply concentrated near the boundaries of the probability simplex.

![Generated and held-out feature distributions](figures/feature-distributions.svg)

The dominant multivariate structure is also learned: for example, the strong relation between jet $p_T$ and mass and the anticorrelations among flavour scores remain visible. The largest residuals occur in correlations involving the flavour-tagging outputs.

<table>
  <tr>
    <td><img src="figures/correlation-test.svg" alt="Held-out feature correlation matrix"></td>
    <td><img src="figures/correlation-generated.svg" alt="Generated feature correlation matrix"></td>
    <td><img src="figures/correlation-difference.svg" alt="Generated minus held-out correlation matrix"></td>
  </tr>
  <tr>
    <td align="center">Held-out jets</td>
    <td align="center">Generated jets</td>
    <td align="center">Generated - held-out</td>
  </tr>
</table>

Most importantly, the generated events reproduce the main reconstructed top-mass peak and the broad asymmetric tail over several orders of magnitude. Across five independent trials reported in the thesis, the top-mass comparison produced an average Kolmogorov-Smirnov statistic of $0.058\pm0.021$. The remaining disagreement shows why downstream observables are a stricter validation criterion than marginal feature agreement alone.

![Generated and held-out reconstructed top-quark mass distributions](figures/top-mass-log-scale.svg)

These results support conditional normalizing flows as a promising fast-simulation approach, while showing that flavour-tagging structure and event-level correlations remain important limitations.

## Running the analysis

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open [`notebooks/atlas_jet_conditional_normalizing_flow.ipynb`](notebooks/atlas_jet_conditional_normalizing_flow.ipynb). The notebook expects `data/jets.parquet`; update `DATA_PATH` if the file is elsewhere. The required reconstructed columns are
`eventNumber`, `pt`, `eta`, `phi`, `mass`, and the four `GN2v01_*` probabilities. The truth context uses `ptFromTruthJet`, `etaFromTruthJet`, `phiFromTruthJet`, and `mFromTruthJet`.

If the parquet file is absent, the notebook creates a small schema-compatible toy dataset and switches to a compact two-block, two-epoch smoke test. Toy mode checks the software path only and must not be used for physics conclusions. Full training is computationally intensive and benefits substantially from a CUDA-capable GPU.

## Repository contents

```text
.
├── docs/
│   └── Jiaxun_Lu_Physics_Thesis_2026.pdf
├── figures/                   # selected analysis results
├── notebooks/
│   └── atlas_jet_conditional_normalizing_flow.ipynb
├── CITATION.cff
├── requirements.txt
└── README.md
```

The notebook is the authoritative implementation; earlier exploratory notebooks are intentionally excluded.

## Limitations and outlook

The fixed six-jet vector imposes an ordering and excludes variable-multiplicity events. Set-, graph-, or transformer-based representations could model variable numbers of jets without a manually chosen ordering. Generating lower-level track and vertex information before applying a dedicated flavour tagger may also reproduce the nearly discrete tagging scores more faithfully.

## References

- J. Lu, [*Normalizing Flows for Detector-Level Jet Generation in Fully Hadronic $t\bar{t}$ Events*](docs/Jiaxun_Lu_Physics_Thesis_2026.pdf), University of Washington undergraduate physics thesis, 2026.
- ATLAS Collaboration, [ATLAS $t\bar{t}$ simulation for ML-based jet flavour tagging (JetSet)](https://opendata.cern.ch/record/atlas-93940), CERN Open Data Portal, 2025. DOI: [10.7483/OPENDATA.ATLAS.QG8W.TO8P](https://doi.org/10.7483/OPENDATA.ATLAS.QG8W.TO8P).
- C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, [“Neural Spline Flows”](https://arxiv.org/abs/1906.04032), NeurIPS 2019.
- V. Stimper et al., [“normflows: A PyTorch Package for Normalizing Flows”](https://doi.org/10.21105/joss.05361), *Journal of Open Source Software* 8, 5361 (2023).

## Author

Jiaxun Lu  
Research advisor: Prof. Gordon Watts
