
import os
import urllib.request as request
import urllib.error

DB_URL = "https://physionet.org/physiobank/database/qtdb/"

EXTS = ['atr', 'dat', 'hea', 'man', 'pu', 'pu0',
        'pu1', 'q1c', 'q2c', 'qt1', 'qt2', 'xws']

def get_ids():
    lines = request.urlopen(os.path.join(DB_URL, "RECORDS")).read().decode("utf-8")
    return lines.split("\n")[:-1]

def download_file(ecg_id, save_dir):
    for ext in EXTS:
        base_name = "{}.{}".format(ecg_id, ext)
        url = os.path.join(DB_URL, base_name)
        try:
            data = request.urlopen(url).read()
            with open(os.path.join(save_dir, base_name), 'wb') as fid:
                fid.write(data)
        except urllib.error.HTTPError as e:
            print(e.msg, url)

if __name__ == "__main__":
    save_dir = "/deep/group/med/qtdb"
    ecg_ids = get_ids()
    for ecg_id in ecg_ids[-10:]:
        print(ecg_id)
        download_file(ecg_id, save_dir)
