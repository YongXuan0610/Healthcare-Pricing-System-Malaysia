# Calibration Summary

The raw quantile intervals were evaluated before calibration on a person-disjoint selection-validation set. Coverage was 88.24% for the nominal 80% candidate and 94.04% for the nominal 90% candidate, with zero quantile crossing.

Final quantile models were fitted on 65,406 episodes from 59,397 people. A separate 14,120-episode, 12,728-person calibration partition used nonnegative conformalized-quantile scores and the maximum score across each person’s episodes. The finite-sample ranks used the requested nominal coverage. Both selected calibration statistics were INR 0.00 because the raw bands were already conservative; no attractive target coverage was manufactured by widening on the locked test.

The 13,985-episode independent audit achieved 88.43% episode coverage and 87.88% simultaneous person coverage for the nominal 80% interval. The corresponding nominal 90% results were 93.82% and 93.57%. The locked test was not used for stopping selection or calibration.

Nonnegative handling clips learned quantile predictions and final lower bounds at zero. A conservative monotone envelope is defined for crossing, but observed crossing was zero. Displayed production bounds may widen only to contain the preserved central point estimate.
