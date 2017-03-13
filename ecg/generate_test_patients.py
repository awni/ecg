from __future__ import print_function
import os
import fnmatch
from tqdm import tqdm


NUM_BUCKETS = 100


def get_bucket_from_id(pat):
    return int(int(pat, 16) % NUM_BUCKETS)


def patient_id(record):
    return os.path.basename(record).split("_")[0]


def get_all_records(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.ecg'):
            yield(os.path.join(root, filename))


def generate_test_patients():
    pids = {}
    record_count = 0
    for record in tqdm(get_all_records('./data/batches')):
        pid = patient_id(record)
        bucket = get_bucket_from_id(pid)
        if bucket < 1:
            pids[pid] = True
            record_count += 1
    print('Patients: ', len(pids.keys()))
    print('Records:  ', record_count)
    print("_____Patient__keys__start__below_____")
    for key in pids.keys():
        print(key)


if __name__ == '__main__':
    generate_test_patients()
