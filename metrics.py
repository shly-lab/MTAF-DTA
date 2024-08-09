import numpy as np
from sklearn.metrics import (accuracy_score,auc,average_precision_score,f1_score,
                                precision_recall_curve,precision_score,recall_score,
                                roc_auc_score,roc_curve,classification_report,r2_score,
                                explained_variance_score)
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

def evaluate_regression(y_true,y_pred):
    metric_dict = {}
    metric_dict['y_true'] = y_true.view(-1)
    metric_dict['y_pred'] = y_pred.view(-1)
    metric_dict['r2'] = r2_score(metric_dict['y_true'], metric_dict['y_pred'],)
    metric_dict['mse'] =  ((metric_dict['y_true']- metric_dict['y_pred'])**2).mean()
    pr,pr_p_val = pearsonr(metric_dict['y_true'], metric_dict['y_pred'])
    metric_dict['pearsonr'] = pr
    metric_dict['pearsonr_p_val'] = pr_p_val
    # metric_dict['concordance_index']= concordance_index(metric_dict['y_true'], metric_dict['y_pred'])
    metric_dict['explained_variance'] = explained_variance_score(metric_dict['y_true'], metric_dict['y_pred'])
    metric_dict['cindex'] = -1 #get_cindex(metric_dict['y_true'], metric_dict['y_pred'])
    metric_dict['rm2'] = get_rm2(metric_dict['y_true'].reshape(-1), metric_dict['y_pred'].reshape(-1))
    return metric_dict
def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))


def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)