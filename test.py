import numpy as np
np.random.seed(42)

from asr_stat_significance import StatisticalSignificance

si_obj = StatisticalSignificance(
    file_path="wer_fleurs.txt", 
    total_batch=1000,
    use_gaussian_appr=True,
    sep='|',
)
ci_obj  = si_obj.compute_significance(
                    num_samples_per_batch=30, confidence_level=0.99)
print(ci_obj)
print(f"The difference in WER between Model X and Y is significant: ", {ci_obj.is_significant()})
