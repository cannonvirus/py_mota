import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import motmetrics as mm

def answer_dataloader(path : str) -> pd.DataFrame:
    
    df = pd.read_csv(path, sep=",")
    df = df[["file_path", "obj_id", "cx", "cy", "width", "height", "radian"]]
    df["file_path"] = df["file_path"].apply(lambda x : re.sub(os.path.dirname(x), "", x))
    df["file_path"] = df["file_path"].apply(lambda x : re.sub("/", "", x))
    df["file_path"] = df["file_path"].apply(lambda x : re.sub(".jpg", "", x))
    df.rename(columns={"file_path" : "frame_num"}, inplace=True)
    df = df.astype({"frame_num" : "int"})    
    df["frame_num"] = df["frame_num"].apply(lambda x : x // 8)
    
    print(df.head())
    
    return df

def target_dataloader(path : str) -> pd.DataFrame:
    
    df = pd.read_csv(path, sep=",")
    df = df[["frame_num", "obj_id", "cx", "cy", "width", "height", "radian"]]
    df = df.astype({"cx" : "float", "cy" : "float"}) 
    
    print(df.head())
    return df

def calc_mota(df_answer : pd.DataFrame, df_target : pd.DataFrame):
    
    max_frame = df_answer["frame_num"].max()
    acc = mm.MOTAccumulator(auto_id=True)
    
    for frm in tqdm(range(max_frame)):
        answer_frame_info = df_answer.loc[df_answer.frame_num == frm, ["cx", "cy"]]
        target_frame_info = df_target.loc[df_target.frame_num == frm, ["cx", "cy"]]
        cost_matrix = mm.distances.norm2squared_matrix(answer_frame_info.to_numpy(), target_frame_info.to_numpy(), max_d2 = 10.)
        # print(cost_matrix)
        
        acc.update(
            df_answer.loc[df_answer.frame_num == frm, ["obj_id"]].to_numpy().flatten(),
            df_target.loc[df_target.frame_num == frm, ["obj_id"]].to_numpy().flatten(),
            cost_matrix
        )
        
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
    
    strsummary = mm.io.render_summary(
        summary,
        formatters={'mota' : '{:.2%}'.format},
        namemap={'mota': 'MOTA', 'motp' : 'MOTP'}
    )
    print(strsummary)
        

if __name__ == "__main__":
    
    df_answer = answer_dataloader(path = "/works/py_mota/room1_000_confirm.txt")
    df_target = target_dataloader(path = "/works/py_mota/room1_000_confirm_tracked.txt")
    
    if df_answer["frame_num"].max() == df_target["frame_num"].max():
        calc_mota(df_answer, df_target)