#import "../common.typ": *

= The product: a small surrogate-modelling toolkit <sec-product>

Before comparing models, we needed a way to compare them _fairly_ and _repeatably_. Early on,
every experiment was a separate script with its own data loading, its own split and its own error
formula, and results could not be compared. We therefore turned the project into a Python package,
`me_project_speit`, with one command-line tool, `me-project-speit`.

#figure(
  {
    set text(8.5pt)
    let box_(t, s, c: soft) = rect(fill: c, radius: 3pt, inset: 6pt, width: 100%, stroke: 0.5pt + accent)[
      #align(center)[#text(weight: "bold")[#t] \ #s]]
    let arr = align(center + horizon, text(14pt, fill: accent)[#sym.arrow.r])
    grid(columns: (1fr, 0.25fr, 1fr, 0.25fr, 1fr, 0.25fr, 1fr), align: horizon,
      box_("generate", [EasyFEA simulations, \ in parallel, seeded]), arr,
      box_("dataset", [CSV + JSON spec \ (inputs, mesh, kind)]), arr,
      box_("train", [compatibility check, \ split, scaling, \ early stopping]), arr,
      box_("infer / plot", [errors per simulation, \ 3D viewer]),
    )
  },
  caption: [The pipeline. Each arrow is one command of `me-project-speit`.],
) <fig-pipeline>

The main design choices, and the problem each one solves:

- *A dataset describes itself.* Every `name.csv` comes with a `name.json` (the _spec_): input names,
  ranges, number of outputs, mesh identifier and _kind_ (`field`, `pointwise` or `graph`, see
  @sec-data). Mixing a dataset and a model that do not fit (for example a 560-output MLP on a
  sponge dataset) is caught _before_ training, with a message listing the models that do fit.
- *A model declares what it accepts.* Models are registered by name (`mlp`, `coord_mlp`, `pod`,
  `gnn`, `gno`) and built from the spec, so the same code adapts to any number of inputs or nodes.
  Adding a model is one file (tutorials in `docs/`).
- *Reproducibility.* Every simulation gets its own random seed derived from the dataset seed, so a
  dataset is identical whatever the number of parallel workers. Checkpoints store the model, its
  arguments, the scaling statistics and the exact split.
- *Honest splitting.* Data is split 70 / 15 / 15 % _by simulation_, never by row. With one row per
  node, a row split would put nodes of the same simulation in training and test, and the test score
  would be far too optimistic.
- *Honest evaluation* (`infer`): errors per simulation with confidence intervals, an independent test
  set made with another seed, and a warning when test parameters are outside the training ranges.
- *Looking at the fields*: a PyVista 3D viewer shows true, predicted and error fields side by side,
  with a cutting plane to see inside the part.

The package has 95 automated tests (`pytest`) that run without the FEM solver. Simulations use the
open-source FEM library *EasyFEA* and the mesher *gmsh*; models use *PyTorch* and *scikit-learn*.

