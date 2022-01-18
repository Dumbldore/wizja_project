import collections
import math
import statistics

import numpy as np
import pandas
from scipy import stats
from scipy.stats import iqr


def raw_vel(curr_deque):
    return curr_deque[-1]


def raw_angle(curr_deque):
    return curr_deque[-1]


def calc_avg_vel(curr_deque):
    return statistics.mean(curr_deque)


def calc_vel_reject_outliers_MAD1(curr_deque):
    """
    Działa ok z max_deviation = 0.9
    """
    max_deviation = 0.9
    u = np.mean(curr_deque)
    s = np.std(curr_deque)
    filtered = [
        e for e in curr_deque if (u - max_deviation * s <= e <= u + max_deviation * s)
    ]
    return np.mean(filtered) if filtered != [] else np.mean(curr_deque)


def calc_vel_reject_outliers_MAD2(curr_deque):
    """
    the best for max_deviation = 0.9 (faster than MAD1)
    """
    max_deviation = 0.9
    data = np.array(curr_deque)
    filtered_data = data[abs(data - np.mean(data)) <= max_deviation * np.std(data)]
    return np.mean(filtered_data) if filtered_data != [] else np.mean(curr_deque)


def calc_vel_reject_outliers_MAD3(curr_deque):
    """
    Jest bardzo spoko z max_deviation = 1.5
    """
    """
    Total error is a bit smaller than for MAD2, this method is better in reducing huge errors but
    is a slightly worse in reducing small errors than MAD2
    """
    max_deviation = 1.5
    data = np.array(curr_deque)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    filtered_data = data[s < max_deviation]
    return np.mean(filtered_data)


def calc_vel_reject_outliers_MAD4(curr_deque):
    """
    Gives the same results as MAD3 but probably is less vulnerable for errors
    """
    max_deviation = 1.5
    data = np.array(curr_deque)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 1.0
    filtered_data = data[s < max_deviation]
    return np.mean(filtered_data)


def calc_vel_reject_outliers_IQR1(curr_deque):
    """
    Jest spoko z iq_range = 0.5 lub 0.4
    :param curr_deque:
    :return:
    """
    sr = pandas.Series(curr_deque)
    iq_range = 0.5
    low_centile = (1 - iq_range) / 2
    high_centile = 1 - low_centile
    qlow, median, qhigh = sr.dropna().quantile([low_centile, iq_range, high_centile])
    iqr = qhigh - qlow
    filtered_data = sr[(sr - median).abs() <= iqr]
    result = np.mean(filtered_data)
    # if len(curr_deque) == 30:
    #     print()
    return result if not math.isnan(result) else median


def calc_vel_reject_outliers_IQR2(curr_deque):
    """
    To słabo działa, daje rezultaty mniej więcej ok, ale struktura jest podejrzana
    """
    axis = 0
    bar = 1.5
    iq_range = 0.5
    low_centile = (1 - iq_range) / 2 * 100
    high_centile = 100 - low_centile
    data = np.array(curr_deque)
    d_iqr = iqr(data, axis=axis)
    d_q1 = np.percentile(data, low_centile, axis=axis)
    d_q3 = np.percentile(data, high_centile, axis=axis)
    iqr_distance = np.multiply(d_iqr, bar)

    stat_shape = list(data.shape)
    if isinstance(axis, collections.Iterable):
        for single_axis in axis:
            stat_shape[single_axis] = 1
    else:
        stat_shape[axis] = 1

    upper_range = d_q3
    upper_outlier = np.less(data - upper_range.reshape(stat_shape), 0)

    lower_range = d_q1
    lower_outlier = np.greater(data - lower_range.reshape(stat_shape), 0)

    iqr_outlier = np.logical_and(upper_outlier, lower_outlier)
    filtered_data = data[iqr_outlier]
    return np.mean(filtered_data) if filtered_data != [] else np.mean(curr_deque)


def calc_vel_reject_outliers_IQR3(curr_deque):
    """
    Dla factor = 1.2 i iq_range = 0.4 jest ok
    """
    factor = 1.2
    iq_range = 0.4
    low_centile = (1 - iq_range) / 2 * 100
    high_centile = 100 - low_centile
    data_in = np.array(curr_deque)
    quant3, quant1 = np.percentile(data_in, [high_centile, low_centile])
    iqr_value = quant3 - quant1
    iqr_sigma = iqr_value / 1.34896
    med_data = np.median(data_in)
    data_out = [
        x
        for x in data_in
        if ((x > med_data - factor * iqr_sigma) and (x < med_data + factor * iqr_sigma))
    ]
    return np.mean(data_out) if data_out != [] else np.mean(curr_deque)


def calc_vel_reject_outliers_IQR4(curr_deque):
    """
    Dla outlierConstant = 0.4 i iq_range = 0.4 jest ok
    """
    outlierConstant = 0.4
    iq_range = 0.4
    low_centile = (1 - iq_range) / 2 * 100
    high_centile = 100 - low_centile

    data = np.array(curr_deque)
    upper_quartile = np.percentile(data, high_centile)
    lower_quartile = np.percentile(data, low_centile)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    result = data[np.where((data >= quartileSet[0]) & (data <= quartileSet[1]))]
    return np.mean(result) if result != [] else np.mean(curr_deque)


def calc_vel_reject_outliers_zscore(curr_deque):
    """
    Jest spoko dla bar = 1.5 i bardzo spoko dla bar - 1.3
    """
    bar = 1.3
    data = np.array(curr_deque)
    d_z = stats.zscore(data, axis=None)
    # filtered_data = np.abs(d_z)
    # filtered_data2 = np.abs(d_z) > bar
    filtered_data = data[np.abs(d_z) < bar]
    results = filtered_data if filtered_data != [] else data
    return np.mean(results)


def calc_avg_angle(curr_deque):
    return statistics.mean(curr_deque)


def calc_angle_reject_outliers_MAD1(curr_deque):
    """
    Działa ok z max_deviation = 3
    """
    max_deviation = 3
    u = np.mean(curr_deque)
    s = np.std(curr_deque)
    filtered = [
        e for e in curr_deque if (u - max_deviation * s <= e <= u + max_deviation * s)
    ]
    return np.mean(filtered) if filtered != [] else np.mean(curr_deque)


def calc_angle_reject_outliers_MAD2(curr_deque):
    """
    To samo co MAD1 ale szybsze chyba
    """
    """
    the best for max_deviation = 3
    """
    max_deviation = 3
    data = np.array(curr_deque)
    filtered_data = data[abs(data - np.mean(data)) <= max_deviation * np.std(data)]
    return np.mean(filtered_data) if filtered_data != [] else np.mean(curr_deque)


def calc_angle_reject_outliers_MAD3(curr_deque):
    """
    Jest spoko z max_deviation = 6
    """
    """
    Total error is a bit smaller than for MAD2, this method is better in reducing huge errors but
    is a slightly worse in reducing small errors than MAD2
    """
    max_deviation = 6
    data = np.array(curr_deque)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    filtered_data = data[s < max_deviation]
    return np.mean(filtered_data)


def calc_angle_reject_outliers_MAD4(curr_deque):
    """
    Jest spoko z max_deviation = 6
    """
    """
    Gives the same results as MAD3 but probably is less vulnerable for errors
    """
    max_deviation = 6
    data = np.array(curr_deque)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 1.0
    filtered_data = data[s < max_deviation]
    return np.mean(filtered_data)


def calc_angle_reject_outliers_IQR1(curr_deque):
    """
    Jest ok z iq_range = 0.5
    :param curr_deque:
    :return:
    """
    sr = pandas.Series(curr_deque)
    iq_range = 0.5
    low_centile = (1 - iq_range) / 2
    high_centile = 1 - low_centile
    qlow, median, qhigh = sr.dropna().quantile([low_centile, iq_range, high_centile])
    iqr = qhigh - qlow
    filtered_data = sr[(sr - median).abs() <= iqr]
    result = np.mean(filtered_data)
    # if len(curr_deque) == 30:
    #     print()
    return result if not math.isnan(result) else median


def calc_angle_reject_outliers_IQR2(curr_deque):
    """
    To słabo działa, daje rezultaty mniej więcej ok, ale struktura jest podejrzana
    """
    axis = 0
    bar = 1.5
    iq_range = 0.5
    low_centile = (1 - iq_range) / 2 * 100
    high_centile = 100 - low_centile
    data = np.array(curr_deque)
    d_iqr = iqr(data, axis=axis)
    d_q1 = np.percentile(data, low_centile, axis=axis)
    d_q3 = np.percentile(data, high_centile, axis=axis)
    iqr_distance = np.multiply(d_iqr, bar)

    stat_shape = list(data.shape)
    if isinstance(axis, collections.Iterable):
        for single_axis in axis:
            stat_shape[single_axis] = 1
    else:
        stat_shape[axis] = 1

    upper_range = d_q3
    upper_outlier = np.less(data - upper_range.reshape(stat_shape), 0)

    lower_range = d_q1
    lower_outlier = np.greater(data - lower_range.reshape(stat_shape), 0)

    iqr_outlier = np.logical_and(upper_outlier, lower_outlier)
    filtered_data = data[iqr_outlier]
    return np.mean(filtered_data) if filtered_data != [] else np.mean(curr_deque)


def calc_angle_reject_outliers_IQR3(curr_deque):
    """
    Dla factor = 2.4 i iq_range = 0.6 jest ok
    """
    factor = 2.4
    iq_range = 0.6
    low_centile = (1 - iq_range) / 2 * 100
    high_centile = 100 - low_centile
    data_in = np.array(curr_deque)
    quant3, quant1 = np.percentile(data_in, [high_centile, low_centile])
    iqr_value = quant3 - quant1
    iqr_sigma = iqr_value / 1.34896
    med_data = np.median(data_in)
    data_out = [
        x
        for x in data_in
        if ((x > med_data - factor * iqr_sigma) and (x < med_data + factor * iqr_sigma))
    ]
    return np.mean(data_out) if data_out != [] else np.mean(curr_deque)


def calc_angle_reject_outliers_IQR4(curr_deque):
    """
    Dla outlierConstant = 3 i iq_range = 0.6 jest ok
    """
    outlierConstant = 3
    iq_range = 0.6
    low_centile = (1 - iq_range) / 2 * 100
    high_centile = 100 - low_centile

    data = np.array(curr_deque)
    upper_quartile = np.percentile(data, high_centile)
    lower_quartile = np.percentile(data, low_centile)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    result = data[np.where((data >= quartileSet[0]) & (data <= quartileSet[1]))]
    return np.mean(result) if result != [] else np.mean(curr_deque)


def calc_angle_reject_outliers_zscore(curr_deque):
    """
    Jest bardzo spoko dla bar = 3
    """
    bar = 3
    data = np.array(curr_deque)
    d_z = stats.zscore(data, axis=None)
    # filtered_data = np.abs(d_z)
    # filtered_data2 = np.abs(d_z) > bar
    filtered_data = data[np.abs(d_z) < bar]
    results = filtered_data if filtered_data != [] else data
    return np.mean(results)
