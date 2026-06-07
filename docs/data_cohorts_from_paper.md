> Raw paper-methods dump. For a structured per-lab summary of acquisition &
> preprocessing for the labs we train on (2, 3, 5), see
> [`lab_measurement_preprocessing_2_3_5.md`](lab_measurement_preprocessing_2_3_5.md).

Methods
Data collection overview
We collected mouse EEG and EMG data from seven different laboratories, resulting in eight cohorts labeled A to H (Table 1). Cohorts
A to E were used to train three automatic classifiers (UsleepEEG,
UsleepEMG, UsleepEEG,EMG) for sleep stage classification. Cohort F was
used to train an L1-penalized logistic regression classifier for NT1
probability estimation, while cohorts G to H served as held-out
datasets for external validation of the pipeline. Each sleep recording includes at least one EEG and one EMG channel, with electrode placement detailed in Table 1. The recording lengths vary
across mice and some mice have multiple recordings. For cohorts
A to F, epochs were manually annotated by trained experts from
the corresponding sleep laboratories as either one of the three
states: wakefulness, NREM sleep, or REM sleep. Since the automatic classifiers are trained on healthy mice, they can only detect
wakefulness, NREM sleep, and REM stages. Consequently, when
Figure 1. A fully automated pipeline for probability estimation of NT1 in mice from unlabeled EEG and EMG data. Flowchart of pipeline. In the
first step, raw EEG and EMG data are fed into two automatic classifiers UsleepEEG and UsleepEMG. Each classifier outputs the probability of either
wakefulness, NREM sleep and REM sleep, yielding two sets of predicted hypnograms yEEG and yEMG. Next, for each y and each sleep stage, we extract
16 features. Seven of these features are relative EEG power bands (slow oscillations, slow delta, fast delta, slow theta, fast theta, alpha, and beta).
Additionally, the root mean square reflects the amplitude of the EMG signal. The remaining eight features come from the hypnogram of y and include:
average bout length [s], total time spent in the sleep stage [%], counts of bouts per hour, total time spent in 4-second bouts [%], total time spent in
bouts with duration from 4-s to −32s [%], total time spent in bouts with duration 32-s to −1-min [%], total time spent in bouts with duration 1 min
to 5 min [%] and total time spent in > 5-min bouts [%]. In total, 16 × 3 × 2 features are obtained and are then fed into an NT1 classifier model that
outputs the probability of having NT1.
Downloaded from https://academic.oup.com/sleepadvances/article/6/2/zpaf025/8128143 by Sektion for Okonomi user on 17 September 2025
SLEEPO, 2025, Vol. 6, No. 2 | 3
applied to NT1 mice, the model cannot detect cataplexy or delta
attacks. These epochs were masked during the validation analysis
(Figure 2), but retained in the training of the NT1 classifier, with
the models classifying them as one of the three main stages.
Experimental data acquisition
Details on animal procedures and data collection of cohorts A-H
from the corresponding labs are separately described below. All
animal studies have been approved by the respective national
Table 1. Overview of data collection
Cohort Lab WT DTA EEG EMG Usleep Lasso Test
A 1 10 0 1 ipsilateral
fronto-parietal differential
Neck X
B 2 17 0 2 parietal
2 frontal
Neck X
C 3 23 0 1 parietal
1 frontal
Neck X
D 4 28 0 1 parietal / 1frontal
or
1 cerebellum / 1 frontal
Neck X
E 5 5 0 1 parietal
1 frontal
Neck X
F 3 11 19 1 parietal
1 frontal
Neck X
G 6 0 7 1 Parietal- Interparietal differential Neck X
H 7 8 6 2 parietal
2 frontal
Neck X
All data were downsampled to 128 Hz and had sleep annotations in four second windows. Cohort A-E is used to train UsleepEEG and UsleepEMG, cohort F is used to
train a NT1 classifier, while cohort G-H are used for validation.
Figure 2. Models trained on one modality can be used to predict wakefulness and sleep in WT and NT1 mice. (A) Confusion matrix of UsleepEMG
with row-wise normalization (recall) tested in a held-out test set of 11 WT mice. (B) The log-odds ratio of the confusion matrices of UsleepEMG from
11 WT and 19 NT mice. Blue color indicates a higher likelihood of the event occurring in WT mice and green color indicates a higher likelihood of
the event occurring in NT mice. * Significant difference between WT and NT (CI of the log-odds ratio (LOR) does not include 0). (C) Confusion matrix
of UsleepEEG with row-wise normalization (recall) tested in the held-out WT mice cohort. (D) The log-odds ratio of the confusion matrix from WT
and NT mice for UsleepEEG. (E) Average root mean square for NT (green) and WT (blue) mice and W = Wakefulness, N = NREM sleep and R = REM
sleep. First window includes epochs where the model and the expert agree, the remaining windows are disagreement epochs where manual label
and predicted label are not the same. (F) Bar plot of the average relative delta power across genotypes (G) Bar plot of the average relative theta power
across genotypes.
Downloaded from https://academic.oup.com/sleepadvances/article/6/2/zpaf025/8128143 by Sektion for Okonomi user on 17 September 2025
4 | Sleep Advances, 2025, Vol. 6, No. 2
authorities and carried out according to ethical guidelines
(European Communities Council Directive (86/609/EEC)) and
ARRIVE (Animal Research: Reporting In Vivo Experiments).
Dataset from Cohort A. Cohort A consists of 10 WT male mice
(C57BL/6J background, 15.0 ± 0.4 weeks of age at surgery). The
data and experimental procedures of this cohort have previously been published [14] (WT control group). The study protocol was approved by the Bologna University ethics committee.
For cohort A sleep scoring was performed on 4-second epochs
by expert investigators using a validated semi-automated procedure (SCOPRISM [15]) on raw EEG and EMG data. Investigators
corrected the automated scoring result if needed based on the
visualization of raw EEG and EMG recordings.
Dataset from Cohort B: Cohort B consists of 17 WT male mice
(6–15 weeks of age, C57BL/6JRj background, Janvier Labs, Le
Genest-Saint-Isle, France). The mice underwent intracerebral injections of a viral vector expressing a calcium sensor,
GCaMP6, under control of the HCRT promoter two weeks
prior to EEG/EMG implantation, and an optical fiber had been
implanted just above the lateral hypothalamus intended for
calcium imaging. This data is not used for the present study.
The experimental procedure for EEG/EMG recordings was similar to what has previously been published in [16]. All experimental procedures were approved by the Veterinary Office of
the Canton of Bern, Switzerland (License number BE 45/18). For
cohort B scoring of the different vigilance stages (wakefulness,
NREM sleep, REM sleep) was conducted in 1-second epochs
using custom-written MATLAB scripts.
Dataset from Cohort C and F. Cohort C consists of 23 wildtype
(WT) mice (12 females, 4-15 weeks of age, C57BL/6 background)
and cohort F consists of 11 WT (10 females, 4-15 weeks of age,
C57BL/6 background) and 19 DTA mice (7 females, 10-15 weeks
of age, double transgenic C57BL/6-Tg (Hcrt/tTA; TetO DTA background). Mice were either purchased from Taconic Biosciences
(C57BL6/J6NTac; Ejby, Denmark), Janvier Labs (C57BL/6JRj; Le
Genest-Saint-Isle, France), or bred in-house as part of a transgenic breeding program.
The experimental procedures of these cohorts have previously been published in [17–19]. All experiments were approved
by the Danish Animal Experiments Inspectorate (license #2019-
15-0201-00016). Wakefulness, NREM sleep, and REM sleep were
determined in four seconds epochs according to standard criteria.
Dataset from Cohort D. Cohort D consists of 28 male mice (WT
or TH-Cre mice,12-24 weeks of age at the time of recording;
C57BL/6 background; Janvier Labs or bred in-house). The experimental procedures of this cohort have previously been published
[20] and a subset of the collected data has previously been published [21]. All experiments were approved by the Danish Animal
Experiments Inspectorate. For cohort D sleep state scoring (wakefulness, NREM sleep, and REM sleep) was performed manually
using SleepScore in either one or four-second epochs based on
standard criteria for EEG and EMG recordings with the assistance
of video.
Dataset from Cohort E. Cohort E consists of five WT male mice
(8–26 weeks of age at recording; C57BL/6J genetic background,
three from Charles River Laboratories, Les Oncins, France and two
donated by Pr. Miquel Vila). Experimental procedures and data
from two WT male mice have previously been published in [22].
All experiments were approved by either the Université Claude
Bernard Lyon 1 Ethic Committee (C2EA-055; #DR-2015-42) or the
CELYNE Ethics Research Committee (C2EA-042; APAFIS#20701).
For cohort E, vigilance states were visually scored using a 5-second
sliding window frame and assigned as either wakefulness, NREM
sleep, or REM sleep.
Dataset from Cohort G: Seven NT1 DTA male mice were double
transgenic offspring of Hcrt/tTA mice (C57BL/6-Tg(Hcrt/tTA)/
Yamanaka) and B6.Cg-Tg(tetO DTA) 1Gfi/J mice (JAX #008468).
Both parental strains were from a C57BL/6J genetic background.
Parental strains and offspring used for EEG/EMG recording were
maintained on a diet (Envigo T-7012, 200 DOXycycline) containing DOX (DOX(+) condition) to repress transgene expression until
neurodegeneration was desired. Mice were maintained on normal
chow for six weeks. The data and experimental procedures of this
cohort have previously been published [23]. All experimental procedures were approved by the Institutional Animal Care and Use
Committee at SRI International.
Dataset from Cohort H: Cohort H consists of eight WT males
(males, 15 weeks of age, C57BL/6 background) and six NT1 male
mice (15 weeks of age, double transgenic C57BL/6-Tg (hcrt/
tTA;TetO DTA background)). Mice were transferred from Taconic
Biosciences (Hudson, NY, USA). Detailed information about EEG/
EMG surgery and data acquisition can be found in Sakai et al.
[24]. All experiments were approved by the Stanford University
Administrative Panel on Laboratory Animal Care and were conducted in accordance with the Stanford University Administrative
Panel on Laboratory Animal Care Guidelines (APLAC-#21,646).