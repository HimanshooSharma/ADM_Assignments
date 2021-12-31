import math
import numpy as np
import hmmlearn.hmm as hmm
from colorama import Fore, Style


S = np.array(["HPM", "LPM"])
V = np.array(["Working", "Defective"])
A = np.array([[0.98, 0.02], [0.1, 0.90]])
B = np.array([[0.95, 0.05], [0.80, 0.20]])
pi = np.array([1.0, 0.0])

max_state_probability = (A.dot(pi)).max()
max_emission_probability = (B.dot(pi)).max()

print(f"{Fore.GREEN}ADM Assignment 5 : HMMs \n{Style.RESET_ALL}")

# What is the most likely sequence of parts conditions in the first three steps of our observation?

print(
    f"{Fore.CYAN}What is the most likely sequence of parts conditions in the first three steps of "
    f"our observation?{Style.RESET_ALL}"
)

print(
    f"{Fore.GREEN}The most likely sequence of parts conditions in the first three steps of our observation with "
    f"their respective probabilities are:{Style.RESET_ALL}"
)
probability = max_emission_probability * 1
for _ in range(3):
    print(f"{V[probability.argmax()]} part with probability: {probability:.4f}")
    probability *= max_state_probability * max_emission_probability

# What is the probability of producing three defective parts in a row (trace: defective, defective, defective)
# in the first three steps of our observation?

print(
    f"{Fore.CYAN}What is the probability of producing three defective parts in a row "
    f"(trace: defective, defective, defective) in the first three steps of our observation?{Style.RESET_ALL}"
)

hmm_model = hmm.MultinomialHMM(n_components=2)
hmm_model.startprob_ = pi
hmm_model.transmat_ = A
hmm_model.emissionprob_ = B

# The probability, of producing three defective parts in a row

defective_trace = np.array([[1, 1, 1]])

# score function will Compute the log prob under the model

defective_trace_Log_probability = hmm_model.score(defective_trace)

# as we know x = exp(ln(x))

defective_trace_probability = math.exp(defective_trace_Log_probability)

print(
    f"{Fore.GREEN}The probability of producing three defective parts in a row "
    f"is {defective_trace_probability:.4f}{Style.RESET_ALL}"
)

# What is the probability of the trace (defective, defective, defective) when starting in steady state?

print(
    f"{Fore.CYAN}What is the probability of the trace (defective, defective, defective) when starting in "
    f"steady state?{Style.RESET_ALL}"
)

hmm_steady_state_model = hmm.MultinomialHMM(n_components=2)
hmm_steady_state_model.startprob_ = np.array([0.82, 0.18])
hmm_steady_state_model.transmat_ = A
hmm_steady_state_model.emissionprob_ = B

defective_trace = np.array([[1, 1, 1]])

defective_trace_Log_probability = hmm_steady_state_model.score(defective_trace)

defective_trace_probability = math.exp(defective_trace_Log_probability)

print(
    f"{Fore.GREEN}The probability of producing three defective parts in a row is {defective_trace_probability:.4f}{Style.RESET_ALL}"
)

# What is the most likely path that led to the production of three working parts in a row
# (trace: working, working, working) in the first three steps of our observation?

print(
    f"{Fore.CYAN}What is the most likely path that led to the production of three working parts in a row "
    f"(trace: working, working, working) in the first three steps of our observation?{Style.RESET_ALL}"
)


work_traces_probability = np.array([[0, 0, 0]]).T
work_traces_log_probability, traces_emission = hmm_model.decode(
    work_traces_probability, algorithm="viterbi"
)
print(
    f"{Fore.GREEN}The most likely path that leads to the production of three working parts "
    f"in a row are: {', '.join(map(lambda i: S[i], traces_emission))}"
    f" with the probability : {math.exp(work_traces_log_probability):.4f}{Style.RESET_ALL}"
)

# What is the most likely path for the trace (working, working, working) when starting in steady state?

print(
    f"{Fore.CYAN}What is the most likely path for the trace (working, working, working)"
    f" when starting in steady state?{Style.RESET_ALL}"
)

work_traces_probability = np.array([[0, 0, 0]]).T
work_traces_log_probability, traces_emission = hmm_steady_state_model.decode(
    work_traces_probability, algorithm="viterbi"
)

print(
    f"{Fore.GREEN}The most likely path that leads to the production of three working parts"
    f" in a row when starting in Steady state are:"
    f"in a row are: {', '.join(map(lambda i: S[i], traces_emission))}"
    f" with the probability : {math.exp(work_traces_log_probability):.4f}{Style.RESET_ALL}"
)
