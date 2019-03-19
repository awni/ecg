def digitize():
  import os
  path = 'clean_30sec_recs'
  files = os.listdir(path)
  l = []
  
  for name in files:
    if ".ecg" in name:
      d = dict()
      def makeIntlist():
        import struct
        with open( path +"/"+ name ,"rb") as n:
          while True:
            try :
              l.append(struct.unpack('i', n.read(4))[0])
            except:
              break
        
      makeIntlist()
      import pandas as pd
      d["data"] = l
      df = pd.DataFrame.from_dict(d)
      savepath = "data/"+name[-15:-10]+".csv"
      df.to_csv(savepath, index = False)   
       
if __name__ == '__main__':
    digitize()
