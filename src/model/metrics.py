import numpy as np
import sklearn.metrics as skm
import torch

def print_latex_metrics(metrics):
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    t2m_keys = [f"t2m/R{i}" for i in vals] + ["t2m/MedR"]
    m2t_keys = [f"m2t/R{i}" for i in vals] + ["m2t/MedR"]

    keys = t2m_keys + m2t_keys

    def ff(val_):
        val = str(val_).ljust(5, "0")
        # make decimal fine when only one digit
        if val[1] == ".":
            val = str(val_).ljust(4, "0")
        return val

    str_ = "& " + " & ".join([ff(metrics[key]) for key in keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in keys}
    print(dico)
    print("Number of samples: {}".format(int(metrics["t2m/len"])))
    print(str_)


def all_contrastive_metrics(
    sims, emb=None, threshold=None, rounding=2, return_cols=False, lengths_threshold=None, gt_lengths=None, est_lengths=None
):
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    if lengths_threshold is not None:
        if est_lengths is None:
            # estimate lengths mean and variance using the first k motions for each query as pseudo-relevant
            k = 30
            idxs = torch.argsort(torch.Tensor(sims), dim=1, descending=True)
            idxs = idxs[:, :k] # take the top 10 idxs for each query
            gtl = gt_lengths.unsqueeze(0).expand(len(sims), -1)
            chosen_lengths = gtl.gather(1, idxs)
            est_lengths = torch.stack([chosen_lengths.mean(dim=1), chosen_lengths.std(dim=1)], dim=1)

        rows = gt_lengths.unsqueeze(1).unsqueeze(1).expand(-1, len(sims), -1)
        cols = est_lengths.unsqueeze(0).expand(len(sims), -1, -1)
        cat = torch.cat([rows, cols], dim=2)    # dim 2 contains 3 elements: gt, est_mean, est_std

        # check if true length lays within the estimated mean +- threshold * std
        mask = (cat[:, :, 0] < cat[:, :, 1] + lengths_threshold * cat[:, :, 2]) & (cat[:, :, 0] > cat[:, :, 1] - lengths_threshold * cat[:, :, 2])

        # put the values in sims corresponding to the false positions in mask to -1 (i.e. the lowest possible value)
        sims[~mask] = -1

    t2m_m, t2m_cols = contrastive_metrics(
        sims, text_selfsim, threshold, return_cols=True, rounding=rounding
    )
    m2t_m, m2t_cols = contrastive_metrics(
        sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding
    )

    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols
    return all_m


def contrastive_metrics(
    sims,
    text_selfsim=None,
    threshold=None,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
):
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = np.diag(dists)[:, None]

    if text_selfsim is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        idx = np.argwhere(text_selfsim > real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        # take as GT the minimum score of similar values
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    rows, cols = np.where(np.isclose(sorted_dists, gt_dists, atol=1e-06))  # find column position of GT

    # if there are ties
    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if return_cols:
        return cols2metrics(cols, num_queries, rounding=rounding), cols
    return cols2metrics(cols, num_queries, rounding=rounding)


def all_m2m_retrieval_metrics(sims, labels):
    metrics = {}
    for label_type, label in labels.items():
        # compute mAP and nDCG for each query
        #_, idxs = np.unique(sims.round(decimals=4), axis=0, return_index=True)
        aps = []
        ndcgs = []

        for t, y in zip(label, sims):
            # find indexes of non-duplicates
            _, idxs = np.unique(y.round(decimals=4), return_index=True)
            idxs = idxs[:-1]    # remove the query from the result set
            idxs = np.sort(idxs)
            y = y[idxs]
            y_true = t == label[idxs]
            if y_true.sum() == 0:
                continue

            ap = skm.average_precision_score(y_true, y)
            aps.append(float(ap))
            ndcg = skm.ndcg_score(y_true.int()[None, :], y[None, :])
            ndcgs.append(float(ndcg))

        metrics[f"{label_type}/mAP"] = sum(aps) / len(aps)
        metrics[f"{label_type}/nDCG"] = sum(ndcgs) / len(ndcgs)
    return metrics    


def break_ties_average(sorted_dists, gt_dists):
    # fast implementation, based on this code:
    # https://stackoverflow.com/a/49239335
    locs = np.argwhere(np.isclose(sorted_dists, gt_dists, atol=1e-06))

    # Find the split indices
    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)

    # Compute the result columns
    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    avg_cols = summed_cols / counts
    return avg_cols


def break_ties_optimistically(sorted_dists, gt_dists):
    rows, cols = np.where(np.isclose(sorted_dists, gt_dists, atol=1e-06))
    _, idx = np.unique(rows, return_index=True)
    cols = cols[idx]
    return cols


def cols2metrics(cols, num_queries, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics
