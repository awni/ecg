from tqdm import tqdm 

def digitize():
  import os
  path = 'clean_30sec_recs/'
  files = os.listdir(path)
  
  d = dict()
  for name in tqdm(files):
    l = []
    id = None
    if ".ecg" in name:
      id = name[:-4]
      def makeIntlist():
        import struct
        with open( path + name ,"rb") as n:
          while True:
            try :
              l.append(struct.unpack('i', n.read(4))[0])
            except:
              break
    if id != None:
      for i in range(3):
        import json
        arg = path + id + "_grp"+str(i)+".episodes.json"
        if os.path.exists(arg):
          id2 = arg
      with open(id2, "r") as read_file:
          ann=json.load(read_file)
          ann = ann["episodes"][0]["rhythm_name"]
        
      makeIntlist()
    
    import pandas as pd
    if id:
      print(ann+id)
      d["200 " +ann + " " + id] = l
  df = pd.DataFrame.from_dict(d)
  return df 
    
if __name__ == '__main__':
    df = digitize()
    savepath = "data.csv"
    df.to_csv(savepath, index = False) 
