import json
import multiprocessing
import os
import shlex
from pathlib import Path
from subprocess import run
from typing import List

from markkk.logger import logger

from blur_single_video import check_single_video


def find_face_in_videos_folder(
    infolder: str = "/data/urop/all_videos_final",
) -> List[str]:
    infolder = Path(infolder).resolve()
    assert infolder.is_dir()

    video_filepath_list = []

    for file in os.listdir(str(infolder)):
        if not file.endswith(".mp4"):
            logger.debug(f"Ignore: {file}")
            continue
        filepath = infolder / file
        assert filepath.is_file()
        if check_single_video(str(filepath)):
            video_filepath_list.append(str(filepath))

    return video_filepath_list


def blur_videos_folder(infolder: str = "/data/urop/all_videos_final") -> List[str]:
    infolder = Path(infolder).resolve()
    assert infolder.is_dir()

    video_filepath_list = []

    for file in os.listdir(str(infolder)):
        if not file.endswith(".mp4"):
            logger.debug(f"Ignore: {file}")
            continue
        filepath = infolder / file
        assert filepath.is_file()
        video_filepath_list.append(str(filepath))

    return video_filepath_list


def blur_videos_list(
    video_filepath_list: List[str],
    out_dir: str = "/data/urop/all_videos_final_blurred",
    overwrite: bool = False,
) -> List[str]:
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_dir():
        os.makedirs(str(out_dir))
        logger.debug(f"Created output dir: {out_dir}")

    cmds = []

    for video_filepath in video_filepath_list:
        video_filepath = Path(video_filepath)
        assert video_filepath.is_file()

        video_name = video_filepath.name

        video_out_path = out_dir / video_name

        if not overwrite:
            assert not video_out_path.is_file(), f"{str(video_out_path)} already exists"
        else:
            if video_out_path.is_file():
                logger.warning(f"Will overwrite {video_out_path}")

        # noone_video
        # cmd = f'python noone_video.py --file "{str(video_filepath)}" --out "{str(video_out_path)}" --show 0'
        # DSFD
        # cmd = f'python blur_video.py -i "{str(video_filepath)}" -o "{str(video_out_path)}" --cuda 1'
        # lightweight DSFD
        cmd = f'python /home/markhuang/code/Anonymizing_video_by_lightDSFD/blur_video.py "{str(video_filepath)}" "{str(video_out_path)}" --cuda 1 --trained_model "/home/markhuang/code/Anonymizing_video_by_lightDSFD/weights/light_DSFD.pth"'
        cmds.append(cmd)

    return cmds


def execute_cmd(cmd: str):
    args = shlex.split(cmd)
    completed = run(args)
    return completed


if __name__ == "__main__":
    face_video_list_fp = Path("video_filepath_list.json")
    face_video_list_fp = Path("test_run.json")

    if face_video_list_fp.is_file():
        with face_video_list_fp.open() as f:
            video_filepath_list = json.load(f)
    else:
        video_filepath_list = find_face_in_videos_folder()

        with face_video_list_fp.open(mode="w") as f:
            json.dump(video_filepath_list, f, indent=4)

    cmds = blur_videos_list(video_filepath_list, overwrite=True)

    _x = str(input("Submit job list to multiprocessing pool? (y/n)")).strip()
    if _x.lower() != "y":
        logger.warning("Operation aborted by instruction.")
        raise Exception("Abort")

    pool = multiprocessing.Pool()
    result = pool.map(execute_cmd, cmds)
