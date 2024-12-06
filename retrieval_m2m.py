import torch
torch.set_num_threads(8)

import os
import glob
from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import yaml
from tqdm import tqdm
from src.config import read_config
import src.prepare  # noqa

logger = logging.getLogger(__name__)


def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)


def compute_sim_matrix(model, dataset, keyids, batch_size=256, estimate_motion_length=False):
    import torch
    import numpy as np
    from src.data.collate import collate_text_motion
    from src.model.tmr import get_sim_matrix

    device = model.device

    nsplit = int(np.ceil(len(dataset) / batch_size))
    with torch.inference_mode():
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        all_data_splitted = np.array_split(all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)
        latent_texts = []
        latent_motions = []
        gt_motion_lengths = []
        est_motion_lengths = []
        sent_embs = []
        motion_labels_dict = {'primary_label_idx': [], 'top_level_label_idx': []}
        for data in tqdm(all_data_splitted, leave=False):
            batch = collate_text_motion(data, device=device)

            # Text is already encoded
            text_x_dict = batch["text_x_dict"]
            motion_x_dict = batch["motion_x_dict"]
            sent_emb = batch["sent_emb"]
            motion_length = motion_x_dict["length"]
            motion_labels = batch["motion_labels"]

            for key, value in motion_labels.items():
                if key in motion_labels_dict:
                    motion_labels_dict[key].append(value)

            # Encode both motion and text
            latent_text = model.encode(text_x_dict, sample_mean=True)
            latent_motion = model.encode(motion_x_dict, sample_mean=True)

            # Estimate motion length
            if estimate_motion_length and hasattr(model, "motion_length_regressor"):
                est_motion_length = model.motion_length_regressor(latent_text)
                est_motion_length = model.motion_length_regressor.denormalize(est_motion_length)
                est_motion_lengths.append(est_motion_length)

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)
            gt_motion_lengths.extend(motion_length)

        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs)
        gt_motion_lengths = torch.Tensor(gt_motion_lengths)
        sim_matrix = get_sim_matrix(latent_motions, latent_motions)
    returned = {
        "sim_matrix": sim_matrix.cpu().numpy(),
        "sent_emb": sent_embs.cpu().numpy(),
        "gt_motion_lengths": gt_motion_lengths,
        "est_motion_lengths": torch.cat(est_motion_lengths).cpu() if len(est_motion_lengths) > 0 else None,
        "motion_labels": {k: torch.cat(v).cpu() for k, v in motion_labels_dict.items()}
    }
    return returned

def dump_motion_lengths(res, path, n_queries=50):
    import pandas as pd

    motion_lengths = res["motion_lengths"]
    sim_matrix = res["sim_matrix"]
    
    # argsort sim_matrix (a numpy array) by rows in descending order
    idxs = sim_matrix.argsort(axis=1)[:, ::-1]
    idxs = idxs[:n_queries]
    dfs = []
    for query_id, row_idxs in enumerate(idxs):
        query_results = [motion_lengths[idx] for idx in row_idxs]
        dfs.append(pd.DataFrame({'query': query_id, 'motion_id': list(range(len(row_idxs))), 'length': query_results}))

    dfs = pd.concat(dfs)
    dfs.to_csv(path + '/motion_lenghts_dump.csv', index=False)
    print(f"Dumped motion lengths in {path}/motion_lenghts_dump.csv for the first {n_queries} queries.")

@hydra.main(version_base=None, config_path="configs", config_name="retrieval_m2m")
def retrieval(newcfg: DictConfig) -> None:
    protocol = newcfg.protocol
    device = newcfg.device
    run_dir = newcfg.run_dir
    ckpt_name = newcfg.ckpt
    batch_size = newcfg.batch_size

    assert protocol in ["all", "normal", "nsim", "guo"]

    if protocol == "all":
        protocols = ["normal", "nsim", "guo"]
    else:
        protocols = [protocol]

    # Load config and replace data
    cfg = read_config(run_dir)
    cfg.data = newcfg.data

    dataset_name = cfg.data.test.path.split("/")[-1]

    save_dir = os.path.join(run_dir, f"{dataset_name}_{ckpt_name}_m2m-metrics")

    # if exists and not empty, skip
    inference_files = glob.glob(os.path.join(save_dir, "*.yaml"))
    if newcfg.skip_already_done and os.path.exists(save_dir) and len(inference_files) == 3:
        logger.info(f"Alredy exists: {save_dir}")
        exit(0)

    os.makedirs(save_dir, exist_ok=True)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.load import load_model_from_cfg
    from src.model.metrics import all_m2m_retrieval_metrics, print_latex_metrics

    pl.seed_everything(cfg.seed)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    datasets = {}
    results = {}
    for protocol in protocols:
        # Load the dataset if not already
        if protocol not in datasets:
            if protocol in ["normal", "threshold", "guo"]:
                dataset = instantiate(cfg.data.test, split="test")
                datasets.update(
                    {key: dataset for key in ["normal", "threshold", "guo"]}
                )
            elif protocol == "nsim":
                datasets[protocol] = instantiate(cfg.data.test, split="nsim_test")
        dataset = datasets[protocol]

        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol in ["normal", "threshold"]:
                res = compute_sim_matrix(
                    model, dataset, dataset.keyids, batch_size=batch_size
                )
                # dump_motion_lengths(res, path=run_dir)
                # exit(0)
                results.update({key: res for key in ["normal", "threshold"]})
            elif protocol == "nsim":
                res = compute_sim_matrix(
                    model, dataset, dataset.keyids, batch_size=batch_size
                )
                results[protocol] = res
            elif protocol == "guo":
                keyids = sorted(dataset.keyids)
                N = len(keyids)

                # make batches of 32
                idx = np.arange(N)
                np.random.seed(0)
                np.random.shuffle(idx)
                idx_batches = [
                    idx[32 * i : 32 * (i + 1)] for i in range(len(keyids) // 32)
                ]

                # split into batches of 32
                # batched_keyids = [ [32], [32], [...]]
                results["guo"] = [
                    compute_sim_matrix(
                        model,
                        dataset,
                        np.array(keyids)[idx_batch],
                        batch_size=256, # FIXME: with lower batch size, it crashes
                    )
                    for idx_batch in idx_batches
                ]
        result = results[protocol]

        # Compute the metrics
        if protocol == "guo":
            all_metrics = []
            for x in result:
                sim_matrix = x["sim_matrix"]
                labels = x["motion_labels"]
                metrics = all_m2m_retrieval_metrics(sim_matrix, labels)
                all_metrics.append(metrics)

            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = round(
                    float(np.mean([metrics[key] for metrics in all_metrics])), 2
                )

            metrics = avg_metrics
        else:
            sim_matrix = result["sim_matrix"]
            labels = result["motion_labels"]

            metrics = all_m2m_retrieval_metrics(sim_matrix, labels)

        # print_latex_metrics(metrics)

        protocol_name = protocol
        metric_name = f"{protocol_name}.yaml"
        path = os.path.join(save_dir, metric_name)
        save_metric(path, metrics)

        logger.info(f"Testing done, metrics saved in:\n{path}")


if __name__ == "__main__":
    retrieval()
