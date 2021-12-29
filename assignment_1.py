import numpy as np
from colorama import Fore, Style


class NoStationarySolutionForStepsError(Exception):
    """Raised when the stationary solution wasn't found for the given number of steps."""

    pass


def calculate_pv_after_n_steps(probability_vector, state_transition_matrix, steps):
    for step in range(steps):
        next_step_p_vector = state_transition_matrix.dot(probability_vector[step])
        probability_vector = np.append(
            probability_vector, np.reshape(next_step_p_vector, (1, 2)), axis=0
        )
    return probability_vector[-1]


def reach_stationary_state(probability_vector, state_transition_matrix, steps):
    steady_state_reached = False
    for step in range(steps):
        next_step_p_vector = state_transition_matrix.dot(probability_vector[step])
        if np.equal(probability_vector[-1], next_step_p_vector).all():
            steady_state_reached = True
            return probability_vector[-1], step
        probability_vector = np.append(
            probability_vector, np.reshape(next_step_p_vector, (1, 2)), axis=0
        )
    if not steady_state_reached:
        raise NoStationarySolutionForStepsError


initial_probability_vector = np.array([[1, 0]])

transition_matrix = np.array([[0.98, 0.1], [0.02, 0.9]])

# What is the probability that the machine is in HPM after 8 minutes if one time step represents one minute?

probability_after_8_steps = calculate_pv_after_n_steps(
    initial_probability_vector, transition_matrix, 8
)
print(
    f"{Fore.CYAN}What is the probability that the machine is in HPM after 8 minutes if one time step represents one "
    f"minute?{Style.RESET_ALL}"
)
print(
    f"{Fore.GREEN}Probability that machine is in HPM after 8 minutes/steps is {probability_after_8_steps[0]}{Style.RESET_ALL} "
)

# What is the probability of producing a working part in the next minute?

probability_after_9_steps = calculate_pv_after_n_steps(
    initial_probability_vector, transition_matrix, 9
)
probability_of_hpm = probability_after_9_steps[0]
probability_of_lpm = probability_after_9_steps[1]

probability_of_working_part = probability_of_lpm * 0.8 + probability_of_hpm * 0.95

print(
    f"{Fore.CYAN}What is the probability of producing a working part in the next minute?{Style.RESET_ALL}"
)

print(
    f"{Fore.GREEN}Probability of of producing a working part given LPM {probability_of_lpm} and hpm "
    f"{probability_of_hpm} is {probability_of_working_part} {Style.RESET_ALL} "
)

# How long does the system need to reach a stationary solution?
probability_vector_at_stationary = None
try:
    probability_vector_at_stationary, stationary_solution_step = reach_stationary_state(
        initial_probability_vector, transition_matrix, 300
    )
    print(f"Steps needed to reach a a stationary solution: {stationary_solution_step}")

except NoStationarySolutionForStepsError:
    print("No steady state found for specified steps!")

# What is the probability to be in HPM in steady state?

if probability_vector_at_stationary is not None:
    print(
        f"The probability to be in HPM in steady state is {probability_vector_at_stationary[0]}"
    )
else:
    print("Steady state has not been found yet!")
