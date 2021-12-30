import numpy as np
from colorama import Fore, Style


def ctmc(probability_vector, steps, discretization, state_transition_matrix):

    t_matrix_discrete = np.add(
        np.array([[1, 0], [0, 1]], dtype=float),
        np.multiply(discretization, state_transition_matrix),
    )
    reached_stationary_state = None
    for step in range(steps):
        next_step_p_vector = t_matrix_discrete.dot(probability_vector[step])
        if np.equal(probability_vector[-1], next_step_p_vector).all():
            reached_stationary_state = step
            return probability_vector, reached_stationary_state
        probability_vector = np.append(
            probability_vector, np.reshape(next_step_p_vector, (1, 2)), axis=0
        )

    return probability_vector, reached_stationary_state


print(f"{Fore.GREEN}ADM Assignment 2 : CTMCs \n{Style.RESET_ALL}")

initial_probability_vector = np.array([[0, 1]])

transition_matrix = np.array([[-0.1, 0.02], [0.1, -0.02]])

# What is the probability that the machine is in HPM after 8 minutes for different discretization
# time steps (e.g. 8, 4, 2, 1, 0.5)?

print(
    f"{Fore.CYAN}What is the probability that the machine is in HPM after 8 minutes for different discretization "
    f"time steps (e.g. 8, 4, 2, 1, 0.5)?{Style.RESET_ALL}"
)
pv, _ = ctmc(initial_probability_vector, 8, 8.0, transition_matrix)
print(f"{Fore.GREEN} For Discretization factor 8: {pv[-1][1]}{Style.RESET_ALL}")

pv, _ = ctmc(initial_probability_vector, 8, 4.0, transition_matrix)
print(f"{Fore.GREEN} For Discretization factor 4: {pv[-1][1]}{Style.RESET_ALL}")

pv, _ = ctmc(initial_probability_vector, 8, 2.0, transition_matrix)
print(f"{Fore.GREEN} For Discretization factor 2: {pv[-1][1]}{Style.RESET_ALL}")

pv, _ = ctmc(initial_probability_vector, 8, 1.0, transition_matrix)
print(f"{Fore.GREEN} For Discretization factor 1: {pv[-1][1]}{Style.RESET_ALL}")

pv, _ = ctmc(initial_probability_vector, 8, 0.5, transition_matrix)
print(f"{Fore.GREEN} For Discretization factor 0.5: {pv[-1][1]}{Style.RESET_ALL}")

# What is the probability to be in HPM in steady state for different discretization time steps (e.g.
# 8, 4, 2, 1, 0.5)?
print(
    f"{Fore.CYAN}What is the probability to be in HPM in steady state for different discretization "
    f"time steps (e.g.8, 4, 2, 1, 0.5)? {Style.RESET_ALL}"
)

pv, stationary_step = ctmc(initial_probability_vector, 30, 8.0, transition_matrix)
print(
    f"{Fore.GREEN} For Discretization factor 8: {pv[-1][1]} at step {stationary_step}{Style.RESET_ALL}"
)

pv, stationary_step = ctmc(initial_probability_vector, 100, 4.0, transition_matrix)
print(
    f"{Fore.GREEN} For Discretization factor 4: {pv[-1][1]} at step {stationary_step}{Style.RESET_ALL}"
)

pv, stationary_step = ctmc(initial_probability_vector, 200, 2.0, transition_matrix)
print(
    f"{Fore.GREEN} For Discretization factor 2: {pv[-1][1]} at step {stationary_step}{Style.RESET_ALL}"
)

pv, stationary_step = ctmc(initial_probability_vector, 400, 1.0, transition_matrix)
print(
    f"{Fore.GREEN} For Discretization factor 1: {pv[-1][1]} at step {stationary_step}{Style.RESET_ALL}"
)

pv, stationary_step = ctmc(initial_probability_vector, 600, 0.5, transition_matrix)
print(
    f"{Fore.GREEN} For Discretization factor 0.5: {pv[-1][1]} at step {stationary_step}{Style.RESET_ALL}"
)

# What is the average probability of producing a working part in steady state?

print(
    f"{Fore.CYAN}What is the average probability of producing a working part in steady state?{Style.RESET_ALL}"
)

ap_of_working_part = pv[-1][1] * 0.95 + pv[-1][0] * 0.8

print(
    f"{Fore.GREEN}Average probability of producing a working part in steady state is "
    f"{ap_of_working_part}{Style.RESET_ALL}"
)
