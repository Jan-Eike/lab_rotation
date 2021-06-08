import numpy as np
import tensorflow as tf
from data_loader.loader import DataGenerator


def main(max_no_dtpts,
        min_no_dtpts,
        time_window,
        n_features,
        n_stat_features,
        features,
        late_patients_only,
        horizon0,
        batch_size,
        no_mc_samples):
    data = DataGenerator(no_mc_samples=no_mc_samples,
                         max_no_dtpts=max_no_dtpts,
                         min_no_dtpts=min_no_dtpts,
                         batch_size=batch_size,
                         fast_load=False,
                         to_save=True,
                         debug=True,
                         fixed_idx_per_class=False,
                         features=features)
    a = next(data.next_batch(batch_size, 0, late=late_patients_only,
                                                       horizon0=horizon0))
    #tf.print(a, [a], summarize=1000000)
    for a_i in a:
        print(a_i.shape)
    for num in range(5):
        for i in range(1):
            print("{}: {}".format(i, a[i][num]))
            print("max: {}".format(max(a[i][num])))
        print("{}: {}".format(8, a[8][num]))
    for i, a_i in enumerate(a):
        print("{}: {}".format(i, a_i[0]))
    
    #print(a[0])

if __name__ == "__main__":
    max_no_dtpts = 250  # chopping 4.6% of data at 250
    min_no_dtpts = 40  # helping with covariance singularity
    time_window = 25  # fixed
    n_features = 24  # old data: 44
    n_stat_features = 8  # old data: 35
    features = 'mr_features_mm_labels'
    n_features= 17
    n_stat_features= 8
    features = None
    late_patients_only = False
    horizon0 = False
    batch_size = 128
    no_mc_samples = np.random.randint(8, high=20, size=None, dtype='l')
    main(max_no_dtpts,
        min_no_dtpts,
        time_window,
        n_features,
        n_stat_features,
        features,
        late_patients_only,
        horizon0,
        batch_size,
        no_mc_samples)