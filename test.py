import json
import math
import os
import shutil
import warnings
from multiprocessing import Process, Queue

import tqdm

from algorithms import *
from args import *
from dataloaders import *
from utils import *

warnings.filterwarnings("ignore")

learning_based_methods = {
    "aucc": aucc,
    "racc": racc,
    "fc4": fc4,
    "c4": c4,
    "c5": c5,
    "pcc": pcc,
    "als": als_only,
    "img": img_only,
    "onenet": one_net,
}

traditional_methods = {
    "bright_pixels": bright_pixels,
    "shades_of_grey": shades_of_grey,
    "grey_world": grey_world,
    "white_patch": white_patch,
    "pca_cc": pca_cc,
    "grey_pixels": grey_pixels,
    "grey_index": grey_index,
    "grey_edge": grey_edge,
    "lsrs": lsrs,
}


def write_res(output_dir, res: dict):
    final_res = {}
    for k, v in res.items():
        final_res[k] = v
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'result.json'), 'w') as f:
        json.dump(final_res, f, indent=1)


def _single_thread(world_size: int, local_rank: int, res_queue: Queue,
                   dataset: dict, method: str, output_dir: str, args,
                   **kwargs):
    num = len(dataset)
    should_handle = math.ceil(len(dataset) / world_size)
    start_index = local_rank * should_handle
    end_index = min(start_index + should_handle, num)
    res = {}
    if local_rank == 0:
        indexes = tqdm.trange(start_index, end_index)
    else:
        indexes = range(start_index, end_index)
    calibrated_wps = None
    for i in indexes:
        data = dataset[i]
        image_name = data['scene_name']
        inputs = data['input']
        als_data = data['als']
        inputs = {'input': inputs, 'extra_input': als_data}
        out = learning_based_methods[method](inputs, opt=args)
        if isinstance(out, (list, tuple)):
            estimate_illum, _ = out
        elif out is not None:
            estimate_illum = out
        else:
            continue

        estimate_illum /= estimate_illum[1]

        if args.post_process and calibrated_wps is not None:
            try:
                estimate_illum = post_process_white_point_nearest(
                    estimate_illum, calibrated_wps)
            except:
                print("Post-process failed for image {}".format(image_name))

        tmp = {}
        tmp['estimation'] = estimate_illum.tolist()
        res[image_name] = tmp

        if data['img'] is not None:
            write_white_balanced_image(data['img'],
                                       estimate_illum,
                                       output_dir,
                                       '{}.png'.format(image_name),
                                       gamma=True)
    res_queue.put(res)


def main(
    method,
    args,
    thread_num=4,
):
    assert method in learning_based_methods, "method should in one of {}".format(
        list(learning_based_methods.keys()))
    if method in learning_based_methods:
        output_folder = args.model_basename
    else:
        output_folder = method
    data_path = args.test_dir
    dataset = TestDataset(data_path,
                          img_size=(args.input_size, args.input_size))
    output_dir = os.path.join(args.out_dir, output_folder)
    shutil.rmtree(output_dir, ignore_errors=True)
    res_queue = Queue(maxsize=thread_num)
    processes = []
    for local_rank in range(thread_num):
        process = Process(target=_single_thread,
                          args=(thread_num, local_rank, res_queue, dataset,
                                method, output_dir, args))
        process.start()
    for process in processes:
        process.join()
    res = {}
    for _ in range(thread_num):
        tmp = res_queue.get()
        for k, v in tmp.items():
            res[k] = v
    res = dict(sorted(res.items(), key=lambda x: x[0]))

    write_res(output_dir, res)


if __name__ == '__main__':
    args = parse_func()
    main(args.model_basename, args, 4)
