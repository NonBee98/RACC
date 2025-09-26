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

output_root = "awb_output"


def wirte_cross_validation_results(output_dir, method_name, results):
    mean_ae = 0
    med_ae = 0
    tri_ae = 0
    best_quad = 0
    worst_quad = 0

    for i in range(len(results)):
        res = results[i]
        mean_ae += res["mean_ae"]
        med_ae += res["med_ae"]
        tri_ae += res["tri_mean_ae"]
        best_quad += res["best_quad"]
        worst_quad += res["worst_quad"]

    mean_ae /= len(results)
    med_ae /= len(results)
    tri_ae /= len(results)
    best_quad /= len(results)
    worst_quad /= len(results)

    final_res = {
        "mean_ae": mean_ae,
        "med_ae": med_ae,
        "tri_mean_ae": tri_ae,
        "best_quad": best_quad,
        "worst_quad": worst_quad,
    }
    with open(
        os.path.join(output_dir, method_name + "_cross_validation.json"), "w"
    ) as f:
        json.dump(final_res, f, indent=4)


def write_res(output_dir, res: dict):
    final_res = {}
    angular_errors = np.array([x["angular_error"] for x in res.values()])
    less_4000_cct_erros = np.array(
        [
            x["cct_error"]
            for x in res.values()
            if x["gt_cct"] < 4000 and x["cct_error"] < 5000
        ]
    )
    greater_4000_cct_erros = np.array(
        [
            x["cct_error"]
            for x in res.values()
            if x["gt_cct"] >= 4000 and x["cct_error"] < 5000
        ]
    )
    cct_erors = np.array(
        [x["cct_error"] for x in res.values() if x["cct_error"] < 5000]
    )
    angular_errors.sort()
    less_4000_cct_erros.sort()
    greater_4000_cct_erros.sort()

    mid_index_1 = (len(angular_errors) + 1) // 2 - 1
    mid_index_2 = (len(angular_errors) + 2) // 2 - 1
    quad_num = len(angular_errors) // 4

    mean_ae = angular_errors.mean()
    med_ae = (angular_errors[mid_index_1] + angular_errors[mid_index_2]) / 2
    best_quad = angular_errors[:quad_num].mean()
    worst_quad = angular_errors[-quad_num:].mean()
    quad_1 = angular_errors[quad_num]
    quad_2 = angular_errors[-quad_num]
    tri_mean_ae = (quad_1 + 2 * med_ae + quad_2) / 4
    mean_cct_error = cct_erors.mean()

    final_res["mean_ae"] = mean_ae
    final_res["med_ae"] = med_ae
    final_res["tri_mean_ae"] = tri_mean_ae
    final_res["best_quad"] = best_quad
    final_res["worst_quad"] = worst_quad
    final_res["mean_cct_error"] = mean_cct_error

    if len(less_4000_cct_erros) > 0:
        mid_index_1 = (len(less_4000_cct_erros) + 1) // 2 - 1
        mid_index_2 = (len(less_4000_cct_erros) + 2) // 2 - 1
        quad_num = len(less_4000_cct_erros) // 4

        mean_cct_error_less_4000 = np.array(less_4000_cct_erros).mean()
        mid_cct_error_less_4000 = (
            less_4000_cct_erros[mid_index_1] + less_4000_cct_erros[mid_index_2]
        ) / 2

        final_res["mean_ce_lt_4000"] = mean_cct_error_less_4000
        final_res["mid_ce_lt_4000"] = mid_cct_error_less_4000

    if len(greater_4000_cct_erros) > 0:
        mid_index_1 = (len(greater_4000_cct_erros) + 1) // 2 - 1
        mid_index_2 = (len(greater_4000_cct_erros) + 2) // 2 - 1
        quad_num = len(greater_4000_cct_erros) // 4

        mean_cct_error_greater_4000 = np.array(greater_4000_cct_erros).mean()
        mid_cct_error_greater_4000 = (
            greater_4000_cct_erros[mid_index_1] + greater_4000_cct_erros[mid_index_2]
        ) / 2

        final_res["mean_ce_gt_4000"] = mean_cct_error_greater_4000
        final_res["mid_ce_gt_4000"] = mid_cct_error_greater_4000

    for k, v in res.items():
        final_res[k] = v

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(final_res, f, indent=1)
    return final_res


def _single_thread(
    world_size: int,
    local_rank: int,
    res_queue: Queue,
    dataset: dict,
    method: str,
    output_dir: str,
    args,
    write_img=True,
    **kwargs
):
    le = len(dataset)
    should_handle = math.ceil(len(dataset) / world_size)
    start_index = local_rank * should_handle
    end_index = min(start_index + should_handle, le)
    res = {}
    if local_rank == 0:
        indexes = tqdm.trange(start_index, end_index)
    else:
        indexes = range(start_index, end_index)
    calibrated_wps = load_calibrated_wps(
        "./assets/calibration/honor_wps.json", type="numpy"
    )
    for i in indexes:
        data = dataset[i]
        image_name = data["scene_name"]
        real_illum = data["illum"]
        inputs = data["input"]
        als_data = data["als"]
        if method in traditional_methods:
            out = traditional_methods[method](inputs)
        else:
            inputs = {"input": inputs, "extra_input": als_data}
            out = learning_based_methods[method](inputs, opt=args)
        if isinstance(out, (list, tuple)):
            estimate_illum, chosen_method = out
        elif out is not None:
            estimate_illum = out
            chosen_method = ""
        else:
            continue

        estimate_illum /= estimate_illum[1]

        tmp = {}
        if args.post_process:
            try:
                estimate_illum = post_process_white_point_nearest(
                    estimate_illum, calibrated_wps
                )
            except:
                print("Post-process failed for image {}".format(image_name))

        ae = calc_ang_error(estimate_illum, real_illum)
        try:
            cct_error, pred_cct, gt_cct = calc_cct_error(
                estimate_illum, real_illum, data["cm_28"], data["cm_65"]
            )
            tmp["gt_cct"] = gt_cct
            tmp["pred_cct"] = pred_cct
            tmp["cct_error"] = cct_error
        except:
            tmp["gt_cct"] = -1
            tmp["pred_cct"] = -1
            tmp["cct_error"] = -1
        tmp["gt_wp"] = real_illum.tolist()
        tmp["pred_wp"] = estimate_illum.tolist()
        tmp["angular_error"] = ae
        res[image_name] = tmp

        text = "{:.2f}° {}".format(ae, chosen_method)
        if write_img:
            write_white_balanced_image(
                data["img"],
                estimate_illum,
                output_dir,
                "{}.png".format(image_name),
                gamma=True,
                text=text,
            )
    res_queue.put(res)


def main(method, args, thread_num=4, write_img=True, **kwargs):
    assert (
        method in traditional_methods or method in learning_based_methods
    ), "method should in one of {}".format(
        list(traditional_methods.keys()) + list(learning_based_methods.keys())
    )
    if method in learning_based_methods:
        output_folder = args.model_name
    else:
        if args.cross_validation:
            output_folder = "{}#{}".format(method, args.tag)
        else:
            output_folder = method
    data_path = args.val_dir
    dataset_name = args.val_dataset_name
    dataset = GeneralDataset(
        data_path,
        mode="test",
        img_size=(args.input_size, args.input_size),
        cross_validation=args.cross_validation,
        fold_num=args.fold_num,
        fold_index=args.fold_index,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
    )
    output_dir = os.path.join(output_root, dataset_name, output_folder)
    shutil.rmtree(output_dir, ignore_errors=True)
    res_queue = Queue(maxsize=thread_num)
    processes = []
    for local_rank in range(thread_num):
        process = Process(
            target=_single_thread,
            args=(
                thread_num,
                local_rank,
                res_queue,
                dataset,
                method,
                output_dir,
                args,
                write_img,
            ),
        )
        process.start()
    for process in processes:
        process.join()
    res = {}
    for _ in range(thread_num):
        tmp = res_queue.get()
        for k, v in tmp.items():
            res[k] = v
    res = dict(sorted(res.items(), key=lambda x: x[0]))

    res = write_res(output_dir, res)
    return res


if __name__ == "__main__":
    args = parse_func()
    if args.cross_validation:
        data_path = args.val_dir
        dataset_name = args.val_dataset_name

            # results = []
            # for fold_index in range(args.fold_num):
            #     args.fold_index = fold_index
            #     args = format_args(args)
            #     result = main(args.model_basename, args, 4, write_img=False)
            #     results.append(result)
            # wirte_cross_validation_results(
            #     os.path.join(output_root, dataset_name), args.model_basename, results
            # )

        for method in traditional_methods.keys():
            results = []
            for fold_index in range(args.fold_num):
                args.fold_index = fold_index
                args = format_args(args)
                result = main(method, args, 4, write_img=False)
                results.append(result)
            wirte_cross_validation_results(
                os.path.join(output_root, dataset_name), method, results)

        for method in learning_based_methods.keys():
            args.model_name = method
            args.model_basename = method
            args = format_args(args)
            results = []
            for fold_index in range(args.fold_num):
                args.fold_index = fold_index
                args = format_args(args)
                result = main(method, args, 4, write_img=False)
                results.append(result)
            wirte_cross_validation_results(
                os.path.join(output_root, dataset_name), method, results)
    else:
        main(args.model_basename, args, 4, write_img=False)

        # for method in traditional_methods.keys():
        #     main(
        #         method,
        #         args,
        #         4,
        #         write_img=False)
