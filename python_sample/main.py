import json
import csv
import pandas as pd
import os
from python_sample.predictor import Predictor

data_path = "./python_sample/data"  # The directory of the input data
output_path = "./python_sample/"  # The directory of the output data


def format_result(result):
    rex = {"accusation": [], "articles": []}

    res_acc = []
    for x in result["accusation"]:
        if not (x is None):
            res_acc.append(int(x))
    rex["accusation"] = res_acc

    res_art = []
    for x in result["articles"]:
        if not (x is None):
            res_art.append(int(x))
    rex["articles"] = res_art

    return rex


if __name__ == "__main__":
    user = Predictor()
    cnt = 0


    def get_batch():
        v = user.batch_size
        if not (type(v) is int) or v <= 0:
            raise NotImplementedError

        return v


    def solve(fact):
        result = user.predict(fact)

        for a in range(0, len(result)):
            result[a] = format_result(result[a])

        return result


    for file_name in os.listdir(data_path):
        #inf = open(os.path.join(data_path, file_name), "r",encoding='utf-8')
        inf=pd.read_csv(os.path.join(data_path, file_name),header=0)
        # ouf = open(os.path.join(output_path, file_name), "w",encoding='utf-8')
        #ouf=open(os.path.join(output_path, 'result.csv'), 'w', newline='')
        print(os.path.join(data_path, file_name))

        #filewriter = csv.writer(ouf)

        fact = []
        
        fact=inf['fact']
        result=solve(fact)
'''
        cnt+=len(result)
        for l in fact:
            for k, v in l.items():
                print(k, v)
                filewriter.writerow([k, v])
            
           
        fact = []
    #读取数据集每一行的fact,每batch(500)行solve一次
        for line in inf:
            fact.append(json.loads(line)["fact"])
            if len(fact) == get_batch():
                result = solve(fact)
                cnt += len(result)
                for x in result:
                    print(json.dumps(x), file=ouf)
                fact = []
        #不足batch的solve一次
        if len(fact) != 0:
            result = solve(fact)
            cnt += len(result)
            for x in result:
                print(json.dumps(x), file=ouf)
            fact = []
'''
#ouf.close()
