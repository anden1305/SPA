Stage summary (counts, seconds, hours, proportion)
              count  total_seconds  proportion        hours
stage_label                                                
NREM         808739        3234885    0.422253   898.579167
REM          114958         459828    0.060021   127.730000
Wake         991598        3966324    0.517726  1101.756667

Stage mapping (numeric->label):
  1 -> Wake
  2 -> NREM
  3 -> REM
  4 -> Artifact

Bout statistics (first 10 rows):
   subject  stage stage_label  mean_bout_length_s  median_bout_length_s  bouts_per_hour
0  sub-001      2        NREM           82.558333                  60.0       16.451224
1  sub-001      3         REM           55.719512                  40.0        4.996298
2  sub-001      1        Wake          126.244858                   8.0       15.552500
3  sub-002      1        Wake          142.072860                  12.0       15.504642
4  sub-002      2        NREM           81.927667                  60.0       15.617609
5  sub-002      3         REM           64.612403                  56.0        1.821584
6  sub-003      2        NREM          116.171190                  84.0       13.401153
7  sub-003      3         REM           64.453608                  56.0        4.070705
8  sub-003      1        Wake          136.298715                   4.0       13.065424
9  sub-004      2        NREM           83.656081                  64.0       18.805307

Recording meta (first 10)
   subject  n_rows  total_duration_s  recording_hours
0  sub-001   64831            259323        72.034167
1  sub-002   64845            259379        72.049722
2  sub-003   64836            259343        72.039722
3  sub-004   64801            259203        72.000833
4  sub-005   64088            256351        71.208611
5  sub-006   64801            259203        72.000833
6  sub-007   64801            259203        72.000833
7  sub-008   64801            259203        72.000833
8  sub-009   64840            259359        72.044167
9  sub-010   64801            259203        72.000833