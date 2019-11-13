#!/usr/bin/env python
# coding=utf-8
from aip import AipImageCensor

from pandas.io.json import json_normalize
from pathlib import Path
import pandas as pd
from typing import Optional, Union, AnyStr
from time import sleep
from itertools import chain
from fire import Fire
import sys


APP_ID = '9722985'
API_KEY = 'rOnVUtKjftl9F4qvBKpvXOmw'
SECRET_KEY = 'kWM5YQSkKXBxmrmcMNfR2PokzrvP7rzK'

SCENES = [
    'ocr', 'public', 'politician', 'antiporn', 'webimage', 'disgust',
    'watermark', 'quality'
]

client = AipImageCensor(APP_ID, API_KEY, SECRET_KEY)


def file_content(fp: str) -> AnyStr:
    """ 读取图片 """
    with open(fp, 'rb') as f:
        return f.read()


# """ 调用色情识别接口 """
# result = client.imageCensorUserDefined(get_file_content('example.jpg'))
# print(result)

# """ 如果图片是url调用如下 """
# result = client.imageCensorUserDefined('http://www.example.com/image.jpg')
# print(result)

#  result = client.faceAudit(get_file_content(face))
# pprint(result)
"""
key为要调用的服务类型，取值如下：
1、ocr：通用文字识别
2、public：公众人物识别
3、politician：政治人物识别
4、antiporn：色情识别
5、terror：暴恐识别。
6、webimage：网图OCR识别
7、disgust:恶心图
8、watermark:水印、二维码
"""
# result = client.imageCensorComb(get_file_content(img), SCENES)
# pprint(result)
# result = client.antiPornGif(get_file_content('antiporn.gif'))

# antiporn, face, terror, public, ocr

# data = pd.DataFrame()


def check_image_porn(fp: Path) -> Optional[pd.DataFrame]:
    try:
        result = client.imageCensorComb(file_content(fp), SCENES)
    except Exception as e:
        print(e, file=sys.stderr)
        return None

    if result is None:
        return None
    a, b = result.get('result', None), result.get('result_fine', None)
    if a is None or b is None:
        return None

    result_coarse = json_normalize(a)
    result_coarse = result_coarse.set_index('class_name').T

    result_fine = json_normalize(b)
    result_fine = result_fine.set_index('class_name').T

    result = json_normalize(result).drop(columns=['result', 'result_fine'])
    result['uri'] = str(fp)
    result['sha1'] = fp.stem

    x = pd.concat([result_coarse, result_fine], axis=1).reset_index(drop=True)
    result = pd.concat([result, x], axis=1)
    return result


def check_dir(d: Union[str, Path], sleep_seconds: int = 1) -> pd.DataFrame:
    data = pd.DataFrame()
    for fp in tqdm(chain(Path(d).glob("*.png"), Path(d).glob("*.jpg"))):
        sleep(sleep_seconds)
        record = check_image_porn(fp)
        if record is None or record.empty:
            print(fp, file=sys.stderr)
            continue
        data = pd.concat([data, record], ignore_index=True, sort=False)
        # data = data.append( record, ignore_index = True, sort = False )
        # record.to_csv(sys.stdout, header=False, index=False)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='check images pornity in directory')
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--sleep',
                        type=int,
                        default=1,
                        help='seconds (default 1 second)')
    parser.add_argument('--dest', type=str, default='.')
    args = parser.parse_args()
    source = Path(args.dir).absolute()
    target = Path(args.dest).joinpath(source.stem).with_suffix('.csv')
    data = check_dir(source, args.sleep)
    data.to_csv(target, index=False)
    for _, record in data.iterrows():
        label = record['conclusion']
        src = Path(record['uri'])
        t = Path(args.dest).joinpath(label, src.name)
        src.rename(t)
