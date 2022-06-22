import sys
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from os_module import *
from mota import *


def img2video(img_folder, out_folder, data, fps=30):

    img_path_list = extract_folder(img_folder, ext=".jpg", full_path=True)
    font_path = "/works/py_mota/demo_source/SCDream5.otf"
    font = ImageFont.truetype(font_path, 70)
    background_info_img = Image.open("/works/py_mota/demo_source/room_info.png", "r")
    room_info_img = Image.open("/works/py_mota/demo_source/room6.png", "r")
    activity_default = 13

    #* sample img read ==
    h, w, _= cv2.imread(img_path_list[0]).shape

    out = cv2.VideoWriter(
                os.path.join(
                    out_folder,
                    os.path.basename(img_folder) + ".mp4",
                ),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
        
    track_footprint = {}

    for idx, imgpath in enumerate(img_path_list):

        img = cv2.imread(imgpath)
        ih, iw, _ = img.shape
        
        if ih != h or iw != w:
            img = cv2.resize(img, (w,h))
        
        frm = int(os.path.basename(os.path.splitext(imgpath)[0]))
        frame_data = data.loc[data.frame_num == (frm / 8), ["obj_id", "cx", "cy"]].to_numpy()
        
        # footprint 내역 
        if len(frame_data) != 0:
            for obj_id, cx, cy in frame_data:
                if obj_id not in track_footprint.keys():
                    track_footprint[obj_id] = []
                track_footprint[obj_id].append([cx, cy])
            
        # 점찍기
        # for key, val in track_footprint.items():
        #     color = [(37 * key) % 255, (23 * key) % 255, (17 * key) % 255]
        #     for ccx, ccy in val:
        #         cv2.circle(img, (int(ccx), int(ccy)), 2, color, -1)
        
        # 선그리기
        for key, val in track_footprint.items():
            
            if len(frame_data) == 0:
                before_data = data.loc[data.frame_num == (frm // 8), ["obj_id", "cx", "cy"]].to_numpy()
            else:
                before_data = frame_data
            
            if key in before_data.T[0]:
            
                if len(val) > 1:
                    for foot_idx, _ in enumerate(val):
                        if foot_idx == 0:
                            pass
                        else:
                            intensity = len(val) - foot_idx
                            lightness = 50 + (5 * intensity)
                            if lightness > 100:
                                continue
                            
                            hls = ((17 * key) % 360, lightness, 100)
                            hls_conv = (hls[0]/2, hls[1]*2.55, hls[2]*2.55)
                            rgb = cv2.cvtColor(np.uint8([[hls_conv]]), cv2.COLOR_HLS2RGB)[0][0]
                            
                            bcx, bcy = val[foot_idx-1]
                            acx, acy = val[foot_idx]
                            
                            # 활동량 계산
                            if foot_idx == len(val) -1:
                                activity_default += np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / 40000
                            cv2.line(img, (int(bcx), int(bcy)), (int(acx), int(acy)), [int(rgb[0]), int(rgb[1]), int(rgb[2])], 4)
        
        # 활동량 랜덤 함수
        
        
        # 후처리 font, png overlay
        img_pil = Image.fromarray(img)
        img_pil = img_pil.convert("RGBA")
        img_pil.paste(background_info_img, (0, 1080-background_info_img.size[1]), mask=background_info_img)    
        img_pil.paste(room_info_img, (0, 0), mask=room_info_img)    
        
        draw = ImageDraw.Draw(img_pil)
        draw.text((260, 730),  "19", font=font, fill=(255,255,255,0)) # 마릿수
        draw.text((250, 860),  str(int(activity_default)), font=font, fill=(255,255,255,0)) # 활동량
        draw.text((280, 990),  "23", font=font, fill=(255,255,255,0)) # 무게
        
        img = np.array(img_pil)
        # cv2.imwrite("test.jpg", img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        out.write(img)
        print(" Process : {} | {}".format(str(idx).zfill(4), len(img_path_list)))
        
        # if idx > 20:
        #     break
    
    out.release()

    return True


if __name__ == "__main__":
    
    dataname = "room2_act2"
    
    img_folder = "/works/py_mota/full_img" # 맨 마지막 / 넣지 말기
    # img_folder = "/works/tracker_debugger/output/room2_act2/interval_8_img" # 맨 마지막 / 넣지 말기
    out_path = "./output"
    df_target = target_dataloader(path = f"/works/cpp_bytetrack_standalone/output/{dataname}_confirm_tracked.txt")
    
    img2video(img_folder, out_path, df_target)