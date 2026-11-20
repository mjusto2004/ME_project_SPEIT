#import "../common.typ": *

= Conclusion and next steps <sec-conclusion>

== What we learned

#figure(
  table(columns: 5, align: (left, left, right, left, left),
    [Problem], [Model], [MAE (°C)], [Strength], [Main limit],
    [cube], [mean field], fmt(get("mean", 720).sim_MAE, d: 1), [–], [–],
    [cube], [MLP (cosine lr), 720 sims], fmt(get("mlp_cos", 720).sim_MAE), [simple, generic], [data hungry, fixed mesh],
    [cube], [coordinate MLP, 360 sims], [0.093], [shares weights over nodes], [no notion of geometry],
    [cube], [POD + ridge, 720 sims], fmt(get("pod_ridge", 720).sim_MAE), [tiny, fast], [capped by its features],
    [cube], [*POD + GP*, 720 sims], [*#fmt(get("pod_gp", 720).sim_MAE)*], [600× better, CPU, seconds], [needs ≥ 90 sims, fixed mesh],
    [sponge], [mean temperature], sm("mean", "split"), [–], [–],
    [sponge], [GNN, 4 rounds], sm("gnn", "split"), [any mesh size], [fails on finer meshes],
    [sponge], [GNN, 8 rounds], [*#sm("gnn8", "split")*], [best on training meshes], [worst on finer meshes],
    [sponge], [*GNO*], sm("gno", "split"), [stable across meshes (#sm("gno", "25") °C at `res` 25)], [slight cold bias, cost],
  ),
  caption: [Summary. Cube: 100 independent simulations; sponges: 150-simulation test split.],
) <tab-summary>

The path of the project, in a few sentences:

+ *A neural network can learn a 3D temperature field*, but a plain MLP needs hundreds of simulations
  and gets the overall level wrong more than the shape.
+ *Looking at the structure of the data paid off more than tuning the network.* The fields of the cube
  live in a 3- to 5-dimensional space (POD). Once we used this, a classical Gaussian process beat every
  neural network by three orders of magnitude, on a CPU, with 90 simulations. When the problem allows
  it, this is the method to use.
+ *Varying the geometry changes everything.* No fixed-size model applies; graph models do, and reach
  about 3 °C (5 % of the range).
+ *Being right on the test set is not enough.* Our best model on training-like meshes (GNN-8) is the
  worst one on finer meshes. Only the GNO, built around physical distances rather than mesh links,
  keeps working when the mesh changes. Testing outside the training conditions (finer meshes, new
  values of $h$) was the most informative experiment of the project.
+ *Check the data.* The coarse-mesh reference moves by up to 18 °C at hot spots when the mesh is
  refined, and rare samples contain impossible temperatures. These facts set the floor of every score
  above.

== Next steps

- *Better data for the sponges*: finer meshes (or a mesh-convergence check), an implicit convection term
  to remove the artefacts, and automatic rejection of samples that break physical bounds.
- *Volume-weighted averages in the GNO*, to remove the cold bias on fine meshes; train on mixed
  resolutions.
- *Mesh-independent POD*: interpolate sponge fields on a common grid, then reuse the POD + GP idea,
  which was so strong on the cube.
- *More data and regularisation for the graph models* (all of them overfit 700 sponges), and physics
  in the loss (energy balance).
- *Time*: predict the whole history $T(bold(x), t)$, not only the final field.

