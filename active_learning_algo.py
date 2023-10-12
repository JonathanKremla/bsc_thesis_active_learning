import copy
import math
from math import exp, log
from random import randint, uniform
import numpy as np
import scipy as sc
from scipy.stats import ttest_ind
from strsimpy.levenshtein import Levenshtein
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import multiprocessing as mp

normalized_lev = NormalizedLevenshtein()
levenshtein = Levenshtein()

alphabet = ['A', 'B', 'C', 'D', 'E']
l = len(alphabet)
Y = {-1, 1}


# Define Oracle
def oracle(candidate, goal_word, oracle_threshold):
    t = 1 - normalized_lev.distance(goal_word, candidate)
    return 1 if t >= oracle_threshold else -1


##################################################


# Define proposal generator
def proposal_generator():
    word_length = randint(2, 10)
    word = ''
    for i in range(word_length):
        l = randint(0, len(alphabet) - 1)
        word += alphabet[l]
    return word


def get_next_sample():
    return proposal_generator()


##################################################


# Define Conditional exponential family model
def phi(y, x, theta, goal_word):
    k = cefm(x, y, theta, goal_word)
    a = A(theta, x, goal_word)
    r = exp(k - a)
    return r


def A(theta, x, goal_word):
    t_0 = cefm(x, -1, theta, goal_word)
    t_1 = cefm(x, 1, theta, goal_word)
    return math.log(math.exp(t_0) + math.exp(t_1))


# Conditional exponential family model
def cefm(x, y, theta, goal_word):
    p = norm_spectrum_kernel(theta, x, goal_word)
    return p if y == 1 else 1 - p


##################################################

# define normed k-spectrum-kernel
def feature_map(k, x):
    res = []
    ext_alphabet = extend_alphabet(k)
    for letter in ext_alphabet:
        res.append(x.count(letter))
    return np.array(res)


def extend_alphabet(k):
    if k == 1:
        return alphabet
    extended_alphabet = copy.deepcopy(alphabet)
    for c, letter in enumerate(alphabet):
        for i in range(k - 1):
            extended_alphabet[c] += letter
    return extended_alphabet


def norm_spectrum_kernel(theta, w1, w2):
    k = int(theta)
    fm_x = feature_map(k, w1)
    fm_y = feature_map(k, w2)
    norm = np.linalg.norm([fm_x, fm_y])
    fm_x = fm_x / norm
    fm_y = fm_y / norm
    return 2 * np.dot(fm_x, fm_y)


##################################################

# Metropolis hastings algorithm with a burn in of 50000 (Line:4-7)
def hasting(current_candidate, target_property, theta, goal_word):
    for _ in range(10000):
        next_candidate = get_next_sample()
        alpha = phi(target_property, next_candidate, theta, goal_word) / phi(target_property, current_candidate, theta,
                                                                             goal_word)
        u = uniform(0, 1)
        if u <= alpha:
            current_candidate = next_candidate
    return current_candidate


##################################################

# Minimization step of the algorithm (Line:9)
def minimize(t, weights, properties, candidates, lmda, goal_word):
    def f(x):
        s = 0
        for i in range(1, t + 1):
            p = norm_spectrum_kernel(x, candidates[i - 1], goal_word)
            p = p if properties[i - 1] == 1 else 1 - p
            r = exp(p - 1)
            n = x
            s += weights[i - 1] * log(r) + lmda * abs(n) ** 2
        return s

    bounds = sc.optimize.Bounds(1, np.inf)
    opt = sc.optimize.minimize(f, np.array([1]), bounds=bounds)
    return opt.x


##################################################

# Execution of the learning algorithm
def learning_algo(theta, budget, target_property, goal_word, oracle_threshold):
    properties = []
    weights = []
    thetas = []
    # initialize theta (Line: 1)
    thetas.append(theta)
    candidates = []
    # iterate through budget (Line: 2)
    for t in range(1, budget + 1):
        # generate new candidate (Line: 3)
        x_t = proposal_generator()
        # execute metropolis hastings algorithm (Line: 6)
        candidates.append(hasting(x_t, target_property, thetas[t - 1], goal_word))
        # oracle evaluation (Line: 8)
        properties.append(oracle(goal_word=goal_word, candidate=candidates[t - 1], oracle_threshold=oracle_threshold))
        # weight assignment (Line: 8)
        weights.append(1 / (phi(target_property, candidates[t - 1], thetas[t - 1], goal_word)))
        # minimization of theta (Line: 9)
        thetas.append(minimize(t, weights, properties, candidates, 0.1, goal_word))
    return candidates[budget - 1]


##################################################


if __name__ == '__main__':
    # initial theta value
    theta = 1
    # Set parameters:
    # Fill with desired budget(s) for the learning algorithm
    budgets = [100]
    # Fill with desired threshold(s) of the Oracle evaluation
    thresholds = [0.8]
    # Fill with desired goal words
    words = ['AAAAA', 'A', 'AAAAAAAAAA', 'ABCDE', 'B', 'BBBBB', 'BBBBBBBBBB']
    # decide number of runs per parameter set
    runs = 100
    # decide max number of parallel processes
    max_processes = 6

    for word in words:
        for threshold in thresholds:
            for el in budgets:
                budget = el
                print("---------------------------------------")
                print("Running for algorithm with params:")
                print("goal word = {}".format(word))
                print("Oracle Threshold  = {}".format(threshold))
                print("budget = {}".format(budget))
                print("runs = {}".format(runs))
                res = []
                results = list()

                with mp.Pool(max_processes) as p:
                    args = [(theta, budget, 1, word, threshold) for _ in range(runs)]
                    for result in p.starmap(learning_algo, args):
                        results.append(result)
                count = [0, 0, 0, 0, 0]
                goal_word_count = 0
                data = []
                rdata = []
                len_sum = 0
                avg_dist = 0
                for w in results:
                    len_sum += len(w)
                    d = normalized_lev.distance(w, word)
                    avg_dist += d 
                    data.append(d)
                    if w == word:
                        goal_word_count += 1
                    for c in w:
                        if c == 'A':
                            count[0] += 1
                        if c == 'B':
                            count[1] += 1
                        if c == 'C':
                            count[2] += 1
                        if c == 'D':
                            count[3] += 1
                        if c == 'E':
                            count[4] += 1
                avg_dist = avg_dist / runs
                print("Reached goal word {} times".format(goal_word_count))
                print("Average distance to goal word: {}".format(avg_dist))
                print(
                    "A: {}, B: {}, C: {} , D: {}, E: {}".format(count[0], count[1], count[2], count[3], count[4]))
                print("Average length: {}".format(len_sum / runs))
                print("{}".format(res))
                avg_r_dist = 0
                for _ in range(runs):
                    nw = proposal_generator()
                    rd = normalized_lev.distance(nw, word)
                    avg_r_dist += rd
                    rdata.append(rd)
                print("Average distance for randomly generated words: {}".format(avg_r_dist / runs))
                print(ttest_ind(data, rdata))
