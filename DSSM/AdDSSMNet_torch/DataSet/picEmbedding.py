import json
import pandas as pd

if __name__ == "__main__":

    with open("C:/Users/EDZ/Desktop/embeddings.txt", 'r', encoding='utf8') as fp:
        json_data = json.load(fp)

    picEmbed = []
    for uid in json_data:
        for tuid, tuidEmbed in json_data[uid].items():
            picEmbed.append([uid, tuid, tuidEmbed])

    tuidDF = pd.DataFrame(picEmbed, columns=['subeventid', 'pid', 'tuidEmbed'])

    tuidDF['pid'] = tuidDF['pid'].astype('int64')
    tuidDF['subeventid'] = tuidDF['subeventid'].astype('int64')

    tuidDF.to_excel("./aaaa.xlsx", index=False)