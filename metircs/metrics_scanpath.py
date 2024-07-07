import re

import numpy as np
from scipy.spatial.distance import directed_hausdorff, euclidean

from fastdtw import fastdtw
import editdistance


def scanpath_to_string(scanpath, height, width, Xbins, Ybins, Tbins):
    """
            a b c d ...
        A
        B
        C
        D

        returns Aa
    """
    if Tbins != 0:
        try:
            assert scanpath.shape[1] == 3
        except Exception as x:
            print("Temporal information doesn't exist.")

    height_step, width_step = height // Ybins, width // Xbins
    string = ''
    num = list()
    for i in range(scanpath.shape[0]):
        fixation = scanpath[i].astype(np.int32)
        xbin = fixation[0] // width_step
        ybin = ((height - fixation[1]) // height_step)
        corrs_x = chr(65 + xbin)
        corrs_y = chr(97 + ybin)
        T = 1
        if Tbins:
            T = fixation[2] // Tbins
        for t in range(T):
            string += (corrs_y + corrs_x)
            num += [(ybin * Xbins) + xbin]
    return string, num


def global_align(P, Q, SubMatrix=None, gap=0, match=1, mismatch=-1):
    """
        https://bitbucket.org/brentp/biostuff/src/
    """
    UP, LEFT, DIAG, NONE = range(4)
    max_p = len(P)
    max_q = len(Q)
    score = np.zeros((max_p + 1, max_q + 1), dtype='f')
    pointer = np.zeros((max_p + 1, max_q + 1), dtype='i')

    pointer[0, 0] = NONE
    score[0, 0] = 0.0
    pointer[0, 1:] = LEFT
    pointer[1:, 0] = UP

    score[0, 1:] = gap * np.arange(max_q)
    score[1:, 0] = gap * np.arange(max_p).T

    for i in range(1, max_p + 1):
        ci = P[i - 1]
        for j in range(1, max_q + 1):
            cj = Q[j - 1]
            if SubMatrix is None:
                diag_score = score[i - 1, j - 1] + (cj == ci and match or mismatch)
            else:
                diag_score = score[i - 1, j - 1] + SubMatrix[cj][ci]
            up_score = score[i - 1, j] + gap
            left_score = score[i, j - 1] + gap

            if diag_score >= up_score:
                if diag_score >= left_score:
                    score[i, j] = diag_score
                    pointer[i, j] = DIAG
                else:
                    score[i, j] = left_score
                    pointer[i, j] = LEFT
            else:
                if up_score > left_score:
                    score[i, j] = up_score
                    pointer[i, j] = UP
                else:
                    score[i, j] = left_score
                    pointer[i, j] = LEFT

    align_j = ""
    align_i = ""
    while True:
        p = pointer[i, j]
        if p == NONE: break
        s = score[i, j]
        if p == DIAG:
            # align_j += Q[j - 1]
            # align_i += P[i - 1]
            i -= 1
            j -= 1
        elif p == LEFT:
            # align_j += Q[j - 1]
            # align_i += "-"
            j -= 1
        elif p == UP:
            # align_j += "-"
            # align_i += P[i - 1]
            i -= 1
        else:
            raise ValueError
    # return align_j[::-1], align_i[::-1]
    return score.max()


def scan_match(P, Q, height, width, Xbins=12, Ybins=8, Tbins=0,
               SubMatrix=None, threshold=3.5, GapValue=0, **kwargs):
    """
        ScanMatch
        You need to creat ScanMatchInfo file before hand in the matlab yourself.

        for more information have look at:
            https://seis.bristol.ac.uk/~psidg/ScanMatch/

    """

    def _create_sub_matrix(Xbins, Ybins, threshold):

        mat = np.zeros((Xbins * Ybins, Xbins * Ybins))
        idx_i = 0
        idx_j = 0

        for i in range(Ybins):
            for j in range(Xbins):
                for ii in range(Ybins):
                    for jj in range(Xbins):
                        mat[idx_i, idx_j] = np.sqrt((j - jj) ** 2 + (i - ii) ** 2)
                        idx_i += 1
                idx_i = 0
                idx_j += 1

        max_sub = mat.max()
        return np.abs(mat - max_sub) - (max_sub - threshold)

    try:

        P = np.array(P, dtype=np.float32)
        Q = np.array(Q, dtype=np.float32)

        P, P_num = scanpath_to_string(P, height, width, Xbins, Ybins, Tbins)
        Q, Q_num = scanpath_to_string(Q, height, width, Xbins, Ybins, Tbins)

        if SubMatrix is None:
            SubMatrix = _create_sub_matrix(Xbins, Ybins, threshold)

        score = global_align(P_num, Q_num, SubMatrix, GapValue)
        scale = SubMatrix.max() * max(len(P_num), len(Q_num))

        return score / scale

    except Exception as e:
        print(e)
        return np.nan


def hausdorff_distance(P, Q, **kwargs):
    if not isinstance(P, np.ndarray):
        P = np.array(P, dtype=np.float32)
    elif P.dtype != np.float32:
        P = P.astype(np.float32)

    if not isinstance(Q, np.ndarray):
        Q = np.array(Q, dtype=np.float32)
    elif Q.dtype != np.float32:
        Q = Q.astype(np.float32)

    return max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0])


def euclidean_distance(P, Q, **kwargs):
    if not isinstance(P, np.ndarray):
        P = np.array(P, dtype=np.float32)
    elif P.dtype != np.float32:
        P = P.astype(np.float32)

    if not isinstance(Q, np.ndarray):
        Q = np.array(Q, dtype=np.float32)
    elif Q.dtype != np.float32:
        Q = Q.astype(np.float32)

    if P.shape == Q.shape:
        return np.sqrt(np.sum((P - Q) ** 2))
    return False


def frechet_distance(P, Q, **kwargs):
    """ Computes the discrete frechet distance between two polygonal lines
    Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    P and Q are arrays of 2-element arrays (points)
    """
    if not isinstance(P, np.ndarray):
        P = np.array(P, dtype=np.float32)
    elif P.dtype != np.float32:
        P = P.astype(np.float32)

    if not isinstance(Q, np.ndarray):
        Q = np.array(Q, dtype=np.float32)
    elif Q.dtype != np.float32:
        Q = Q.astype(np.float32)

    def _c(ca, i, j, P, Q):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = euclidean(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(ca, i - 1, 0, P, Q), euclidean(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(ca, 0, j - 1, P, Q), euclidean(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(_c(ca, i - 1, j, P, Q), _c(ca, i - 1, j - 1, P, Q), _c(ca, i, j - 1, P, Q)),
                euclidean(P[i], Q[j]))
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(P) - 1, len(Q) - 1, P, Q)


# 2
def DTW(P, Q, **kwargs):
    dist, _ = fastdtw(P, Q, dist=euclidean)
    return dist


# 5
def TDE(
        P,
        Q,

        # options
        k=3,  # time-embedding vector dimension
        distance_mode='Mean', **kwargs
):
    """
        code reference:
            https://github.com/dariozanca/FixaTons/
            https://arxiv.org/abs/1802.02534

        metric: Simulating Human Saccadic Scanpaths on Natural Images.
                 wei wang etal.
    """

    # P and Q can have different lenghts
    # They are list of fixations, that is couple of coordinates
    # k must be shorter than both lists lenghts

    # we check for k be smaller or equal then the lenghts of the two input scanpaths
    if len(P) < k or len(Q) < k:
        print('ERROR: Too large value for the time-embedding vector dimension')
        return False

    # create time-embedding vectors for both scanpaths

    P_vectors = []
    for i in np.arange(0, len(P) - k + 1):
        P_vectors.append(P[i:i + k])

    Q_vectors = []
    for i in np.arange(0, len(Q) - k + 1):
        Q_vectors.append(Q[i:i + k])

    # in the following cicles, for each k-vector from the simulated scanpath
    # we look for the k-vector from humans, the one of minumum distance
    # and we save the value of such a distance, divided by k

    distances = []

    for s_k_vec in Q_vectors:

        # find human k-vec of minimum distance

        norms = []

        for h_k_vec in P_vectors:
            d = np.linalg.norm(euclidean_distance(s_k_vec, h_k_vec))
            norms.append(d)

        distances.append(min(norms) / k)

    # at this point, the list "distances" contains the value of
    # minumum distance for each simulated k-vec
    # according to the distance_mode, here we compute the similarity
    # between the two scanpaths.

    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)
    else:
        print('ERROR: distance mode not defined.')
        return False


# 3
def REC(P, Q, threshold, **kwargs):
    """
        Cross-recurrence
        https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf

    """

    def _C(P, Q, threshold):
        assert (P.shape == Q.shape)
        shape = P.shape[0]
        c = np.zeros((shape, shape))

        for i in range(shape):
            for j in range(shape):
                if euclidean(P[i], Q[j]) < threshold:
                    c[i, j] = 1
        return c

    P = np.array(P, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]

    c = _C(P, Q, threshold)
    R = np.triu(c, 1).sum()
    return 100 * (2 * R) / (min_len * (min_len - 1))


# 4
def DET(P, Q, threshold, **kwargs):
    """
        https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
    """

    def _C(P, Q, threshold):
        assert (P.shape == Q.shape)
        shape = P.shape[0]
        c = np.zeros((shape, shape))

        for i in range(shape):
            for j in range(shape):
                if euclidean(P[i], Q[j]) < threshold:
                    c[i, j] = 1
        return c

    P = np.array(P, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]

    c = _C(P, Q, threshold)

    R = np.triu(c, 1).sum()

    counter = 0
    for i in range(1, min_len):
        data = c.diagonal(i)
        data = ''.join([str(int(item)) for item in data])  # mark
        counter += len(re.findall('1{2,}', data))

    return 100 * (counter / R)


# 1
def levenshtein_distance(P, Q, height, width, Xbins=12, Ybins=8, **kwargs):
    """
        Levenshtein distance
    """

    P, P_num = scanpath_to_string(P, height, width, Xbins, Ybins, 0)
    Q, Q_num = scanpath_to_string(Q, height, width, Xbins, Ybins, 0)

    return editdistance.eval(P, Q)
