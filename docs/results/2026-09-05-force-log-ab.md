# Selection A/B: forcing The Log, gated on its value map

  loaded model_weights_final_ep83128.pth (episodes_completed=83128, clean_load=True)
card     : The Log (id 33)
arms     : A greedy   |   B force p=1.0 gated at 459.0
opponent : UtilityTeacher rung 10, 16 pool decks
paired   : 160 trials, same seed and deck per trial

    20/160  A 0.500  B 0.400  forced 136
    40/160  A 0.475  B 0.388  forced 357
    60/160  A 0.483  B 0.408  forced 560
    80/160  A 0.500  B 0.394  forced 791
   100/160  A 0.510  B 0.385  forced 1003
   120/160  A 0.500  B 0.404  forced 1227

   160/160  A 0.494  B 0.422  forced 1605

  A greedy       0.4938
  B forced       0.4219
  paired delta   -0.0719   95% CI [-0.1625, +0.0125]
  21 better / 33 worse / 106 tied   sign test p = 0.1337
  forced plays   1605 over 37236 decisions (4.3%)

  VERDICT: forcing The Log no effect resolved
  (n is the binding constraint on this class of comparison -- CLAUDE.md records the control arm alone varying 0.570-0.700 across runs)
