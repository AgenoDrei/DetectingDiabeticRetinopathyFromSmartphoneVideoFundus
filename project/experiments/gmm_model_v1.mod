%YAML:1.0
---
opencv_ml_em:
   format: 3
   training_params:
      nclusters: 6
      cov_mat_type: diagonal
      epsilon: 1.
      iterations: 10
   weights: !!opencv-matrix
      rows: 1
      cols: 6
      dt: d
      data: [ 2.5355006449783990e-01, 1.4934429295150287e-01,
          2.3905238530414563e-01, 1.3065514148071888e-01,
          1.7098992522379702e-01, 5.6408190540730735e-02 ]
   means: !!opencv-matrix
      rows: 6
      cols: 3
      dt: d
      data: [ 3.0759858009828228e+01, 3.7852151978298409e+01,
          9.7968953207766219e+01, 9.8545532284037037e+01,
          5.4158422866426072e+01, 1.7147451531339965e+02,
          1.0517542502206413e+02, 4.7151052123581351e+01,
          7.3813522718485160e+01, 1.7849085528446022e+01,
          7.4741473839401550e+01, 1.7089943097525730e+02,
          2.0632674365933465e+01, 9.9434495527359473e+01,
          8.7459783049603530e+01, 1.0381675569316138e+02,
          1.4566173824050927e+02, 5.9989648598994897e+01 ]
   covs:
      - !!opencv-matrix
         rows: 3
         cols: 3
         dt: d
         data: [ 3.2450795976336730e+02, 0., 0., 0.,
             3.5625165817368935e+02, 0., 0., 0., 1.0583457703095492e+03 ]
      - !!opencv-matrix
         rows: 3
         cols: 3
         dt: d
         data: [ 5.1532316570385808e+02, 0., 0., 0.,
             1.1627667571756338e+03, 0., 0., 0., 1.7822569386763007e+03 ]
      - !!opencv-matrix
         rows: 3
         cols: 3
         dt: d
         data: [ 5.2075388846892520e+02, 0., 0., 0.,
             5.4650089189710593e+02, 0., 0., 0., 1.0438805114984264e+03 ]
      - !!opencv-matrix
         rows: 3
         cols: 3
         dt: d
         data: [ 6.6404627217801973e+01, 0., 0., 0.,
             1.2793484160740072e+03, 0., 0., 0., 1.3963296228002564e+03 ]
      - !!opencv-matrix
         rows: 3
         cols: 3
         dt: d
         data: [ 4.2320901362099860e+01, 0., 0., 0.,
             9.3912324579660878e+02, 0., 0., 0., 6.2035381356374978e+02 ]
      - !!opencv-matrix
         rows: 3
         cols: 3
         dt: d
         data: [ 4.7668747263341214e+02, 0., 0., 0.,
             2.4338048111890034e+03, 0., 0., 0., 1.6746564623619263e+03 ]
