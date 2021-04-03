import json
import multiprocessing
import os
import shlex
from pathlib import Path
from subprocess import run
from typing import List

from markkk.logger import logger

from blur_single_video import check_single_video


def find_face_in_videos_folder(infolder: str = "/data/urop/all_videos_final"):
    infolder = Path(infolder).resolve()
    assert infolder.is_dir()

    video_filepath_list = []

    for file in os.listdir(str(infolder)):
        if not file.endswith(".mp4"):
            logger.debug(f"Ignore: {file}")
            continue
        filepath = infolder / file
        assert filepath.is_file()
        if check_single_video(filepath):
            video_filepath_list.append(filepath)

    return video_filepath_list


def blur_videos_folder(infolder: str = "/data/urop/all_videos_final"):
    infolder = Path(infolder).resolve()
    assert infolder.is_dir()

    video_filepath_list = []

    for file in os.listdir(str(infolder)):
        if not file.endswith(".mp4"):
            logger.debug(f"Ignore: {file}")
            continue
        filepath = infolder / file
        assert filepath.is_file()
        video_filepath_list.append(filepath)

    return video_filepath_list


def blur_videos_list(
    video_filepath_list: List[Path],
    out_dir: str = "/data/urop/all_videos_final_blurred",
    overwrite: bool = False,
) -> List[str]:
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_dir():
        os.makedirs(str(out_dir))
        logger.debug(f"Created output dir: {out_dir}")

    cmds = []

    for video_filepath in video_filepath_list:
        assert video_filepath.is_file()

        video_name = video_filepath.name

        video_out_path = out_dir / video_name

        if not overwrite:
            assert video_out_path.is_file()
        else:
            logger.warning(f"Will overwrite {video_out_path}")

        cmd = f'python noone_video.py --file "{video_filepath}" --out "{video_out_path}" --show 0'
        cmds.append(cmd)

    return cmds


def execute_cmd(cmd: str):
    args = shlex.split(cmd)
    completed = run(args)
    return completed


if __name__ == "__main__":
    face_video_list_fp = Path("video_filepath_list.json")

    if face_video_list_fp.is_file():
        with face_video_list_fp.open() as f:
            video_filepath_list = json.load(f)
    else:
        video_filepath_list = find_face_in_videos_folder()

        with face_video_list_fp.open(mode="w") as f:
            json.dump(video_filepath_list, f, indent=4)

    cmds = blur_videos_list(video_filepath_list)

    _x = str(input("Submit job list to multiprocessing pool? (y/n)")).strip()
    if _x.lower() != "y":
        logger.warning("Operation aborted by instruction.")
        raise Exception("Abort")

    pool = multiprocessing.Pool()
    result = pool.map(execute_cmd, cmds)
