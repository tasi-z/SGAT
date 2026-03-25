import json
import logging
from datetime import datetime

import h5py
import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def load_eval(dir):
    summaries, results = {}, {}
    with h5py.File(str(dir / "results.h5"), "r") as hfile:
        for k in hfile.keys():
            r = np.array(hfile[k])
            if len(r.shape) < 3:
                results[k] = r
        for k, v in hfile.attrs.items():
            summaries[k] = v
    with open(dir / "summaries.json", "r") as f:
        s = json.load(f)
    summaries = {k: v if v is not None else np.nan for k, v in s.items()}
    return summaries, results


def save_eval(dir, summaries, figures, results):
    # detiled_keys = ["epi_prec_detail", "prec_detail","ransac_inl_detail"]
    ignore_keys = ["H_error_affine","max_l","affine_num","est_H","epi_prec_detail", "prec_detail","ransac_inl_detail", "matched_pts0", "matched_pts1"]
    if "est_H" in results:
        np.save(dir / "est_H.npy", results["est_H"])
    if "epi_prec_detail" in results:
        np.save(dir / "epi_prec_detail.npy", np.array(results["epi_prec_detail"], dtype=object))
    if "ransac_inl_detail" in results:
        np.save(dir / "ransac_inl_detail.npy", np.array(results["ransac_inl_detail"], dtype=object))
    if "prec_detail" in results:
        np.save(dir / "prec_detail.npy", np.array(results["prec_detail"], dtype=object))
    if "matched_pts0" in results:
        np.save(dir / "matched_pts0.npy", np.array(results["matched_pts0"], dtype=object))
    if "matched_pts1" in results:
        np.save(dir / "matched_pts1.npy", np.array(results["matched_pts1"], dtype=object))
    with h5py.File(str(dir / "results.h5"), "w") as hfile:
        for k, v in results.items():
            if k in ignore_keys:
                continue
            # if k in detiled_keys:
            #     arr = np.array(v, dtype=object)
            #     dt = h5py.special_dtype(vlen=np.float64)
            #     hfile.create_dataset(k, data=arr, dtype=dt)
            else:
                arr = np.array(v)
                if not np.issubdtype(arr.dtype, np.number):
                    arr = arr.astype("object")
                hfile.create_dataset(k, data=arr)
        # just to be safe, not used in practice
        for k, v in summaries.items():
            hfile.attrs[k] = v
    s = {
        k: float(v) if np.isfinite(v) else None
        for k, v in summaries.items()
        if not isinstance(v, list)
    }
    s = {**s, **{k: v for k, v in summaries.items() if isinstance(v, list)}}
    with open(dir / "summaries.json", "w") as f:
        json.dump(s, f, indent=4)

    for fig_name, fig in figures.items():
        fig.savefig(dir / f"{fig_name}.png")


def exists_eval(dir):
    return (dir / "results.h5").exists() and (dir / "summaries.json").exists()


class EvalPipeline:
    default_conf = {}

    export_keys = []
    optional_export_keys = []

    def __init__(self, conf):
        """Assumes"""
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self._init(self.conf)

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        raise NotImplementedError

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        raise NotImplementedError

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        raise NotImplementedError

    def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False,config=None):
        """Run export+eval loop"""
        # 初始化日志文件
        log_file = experiment_dir / "eval_log.txt"
        start_time = datetime.now()

        def write_log(msg):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{timestamp}] {msg}"
            logger.info(msg)
            with open(log_file, "a") as f:
                f.write(log_msg + "\n")

        write_log("=" * 60)
        write_log(f"Eval Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        write_log(f"Output Dir: {experiment_dir}")
        write_log("=" * 60)

        # 记录主要配置参数
        write_log("Configuration:")
        write_log(f"  Model: {self.conf.model.get('name', 'N/A')}")
        if hasattr(self.conf.model, 'matcher'):
            matcher_conf = self.conf.model.matcher
            write_log(f"  Matcher: {matcher_conf.get('name', 'N/A')}")
            if 'weights' in matcher_conf:
                write_log(f"  Weights: {matcher_conf.weights}")
        if hasattr(self.conf.model, 'extractor'):
            extractor_conf = self.conf.model.extractor
            write_log(f"  Extractor: {extractor_conf.get('name', 'N/A')}")
            write_log(f"  Max Keypoints: {extractor_conf.get('max_num_keypoints', 'N/A')}")
        write_log(f"  Data: {self.conf.data.get('name', 'N/A')}")
        write_log(f"  Estimator: {self.conf.eval.get('estimator', 'N/A')}")
        write_log(f"  RANSAC Threshold: {self.conf.eval.get('ransac_th', 'N/A')}")
        write_log("-" * 60)

        # 保存配置
        write_log("Saving configuration...")
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        write_log("Configuration saved.")

        # 获取预测
        write_log("Getting predictions...")
        pred_start = datetime.now()
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite
        )
        pred_time = (datetime.now() - pred_start).total_seconds()
        write_log(f"Predictions completed in {pred_time:.2f}s")
        write_log(f"Predictions file: {pred_file}")

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            write_log("Running evaluation...")
            eval_start = datetime.now()
            s, f, r = self.run_eval(self.get_dataloader(self.conf.data), pred_file)
            eval_time = (datetime.now() - eval_start).total_seconds()
            write_log(f"Evaluation completed in {eval_time:.2f}s")

            write_log("Saving evaluation results...")
            save_eval(experiment_dir, s, f, r)
            write_log("Results saved.")
        else:
            write_log("Loading existing evaluation results...")

        s, r = load_eval(experiment_dir)

        # 记录评估结果摘要
        write_log("-" * 60)
        write_log("Evaluation Results:")
        for k, v in s.items():
            if isinstance(v, (int, float)):
                write_log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        write_log("-" * 60)
        write_log(f"Total Time: {total_time:.2f}s")
        write_log(f"Eval End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        write_log("=" * 60)

        return s, f, r

    def save_conf(self, experiment_dir, overwrite=False, overwrite_eval=False):
        # store config
        conf_output_path = experiment_dir / "conf.yaml"
        if conf_output_path.exists():
            saved_conf = OmegaConf.load(conf_output_path)
            if (saved_conf.data != self.conf.data) or (
                saved_conf.model != self.conf.model
            ):
                assert (
                    overwrite
                ), "configs changed, add --overwrite to rerun experiment with new conf"
            if saved_conf.eval != self.conf.eval:
                assert (
                    overwrite or overwrite_eval
                ), "eval configs changed, add --overwrite_eval to rerun evaluation"
        OmegaConf.save(self.conf, experiment_dir / "conf.yaml")
