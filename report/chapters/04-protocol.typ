#import "../common.typ": *

= How we measure errors <sec-protocol>

All errors are in °C (a temperature _difference_, so 1 °C = 1 K).

- *MAE per simulation*: the mean absolute error over the nodes of one simulation, then averaged over
  simulations. Each simulation weighs the same, whatever its number of nodes. This is our main score.
- *95 % confidence interval*: obtained by bootstrap, i.e. by re-drawing the test simulations with
  replacement 2000 times. The nodes of one simulation are strongly correlated, so the simulation, not
  the node, is the independent unit. If the intervals of two models overlap, the test set cannot
  separate them.
- *NMAE*: MAE divided by the temperature range (max – min) of the true field. It tells whether an
  error is small compared to what there is to predict.
- *Baseline*: predicting the mean training temperature everywhere. Any useful model must beat it by
  far.

For the cube, every model is scored on the *same 100 independent simulations* (seed 2), never seen
during training or model selection. For the sponges, we use the 150-simulation test split of the
training set and the four extra test sets of @tab-sponge-sets.

