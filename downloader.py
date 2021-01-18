import json
import argparse
import os
import os.path as osp
import shutil
from tqdm import tqdm

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def parseInput():
    parser = argparse.ArgumentParser("msr_vtt_downloader")
    parser.add_argument('--save_path', type=str, default='./MSR_VTT', help='path to save the videos')
    parser.add_argument('--json_file', type=str, default='./videodatainfo_2017.json', help='path to save the json file')
    parser.add_argument('--vid', type=int, default=-1, help='download a specific video with a vid. -1 to download all')
    parser.add_argument('--cid', type=int, default=-1, help='check caption with an id of cid')
    return parser.parse_args()

def download(vid, url, save_path):
    name = vid+".mp4"
    
    if osp.exists(osp.join(save_path, name)):
        return

    os.system("youtube-dl -f mp4 {} >> ./download_log.txt".format(url))
    file = [x for x in os.listdir() if '.mp4' in x][0]
    os.rename(file, name)
    shutil.move(name, osp.join(save_path, name))


def main(config):
    
    if not osp.exists(config.json_file):
        raise RuntimeError("INVALID json file: {}".format(config.json_file))
    
    if not osp.exists(config.save_path):
        os.mkdir(config.save_path)

    json_file = load_json(config.json_file)

    videos = json_file['videos']

    for video in tqdm(videos):
        id = video['id']
        vid = video['video_id']
        # cat = video['category']
        url = video['url']
        # st = video['start time']
        # ed = video['end time']
        # sp = video['split']
        # print("id: {}, vid: {}, url: {}".format(id, vid, url))
        if id == -1:
            download(vid, url, config.save_path)
        elif id == config.vid:
            download(vid, url, config.save_path)
            print("Done")
            break

    captions = json_file['sentences']

    for cap in tqdm(captions):
        cap_id = cap['sen_id']
        vid = cap['video_id']
        cap = cap['caption']
        if config.cid == cap_id:
            print("Captions {}: {}".format(cap_id, cap))

main(parseInput())
