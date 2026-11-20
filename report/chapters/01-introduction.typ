#import "../common.typ": *

= Introduction

== The problem

Engineers often need the temperature inside a part: will a component overheat, where is the hot spot,
how fast does it cool? The standard tool is the *finite-element method* (FEM). The part is cut into
small tetrahedra (the *mesh*), and the heat equation is solved on this mesh, step by step in time.
FEM is accurate, but each run costs time: about one second for our smallest case and tens of seconds
for a fine mesh. In a design loop (optimisation, uncertainty study, real-time tool) we would like to
run it thousands of times.

A *surrogate model* replaces the solver by a fast approximation learned from examples. We run the
solver a few hundred times, store the inputs and the resulting temperature fields, and train a model
to map inputs to fields. Once trained, a prediction takes about a millisecond.

The question of this project is therefore:

#align(center, block(width: 88%)[
  _Can a machine-learning model predict the full 3D temperature field of a part, from its physical
  parameters (and later from its geometry), accurately enough to replace the solver, and with how
  much data?_
])

== Two levels of difficulty

We worked on two problems, the second much harder than the first:

+ *A fixed cube* (@sec-cube). Always the same geometry and the same mesh of 560 nodes. Only three
  physical numbers change: initial temperature $T_0$, heat flux $Phi_0$ and cooling coefficient $h$.
  The output is a vector of 560 temperatures. This is a classic "parameters $->$ field" regression.
+ *Random sponges* (@sec-sponge). A cube with 1 to 20 random spherical holes, heated through a
  random face. Every sample has its *own* mesh (from 271 to 2977 nodes), so the output has no fixed
  size and a node has no fixed meaning. Here the model must understand geometry, not only parameters.

== Path of the report

We present the tool we built (@sec-product), the physics and the datasets (@sec-data), how we
measure errors (@sec-protocol), then each family of models, always in the same order: _why we tried
it, how it works, how well it does, where it breaks_. Framed remarks
explain the reasoning that led from one step to the next; those titled _Limits_ list the limits.

