from termcolor import colored
from collections import deque
import time
import random
class Solution:
    """
    2534. Time Taken to Cross the Door - HARD

    There are n persons numbered from 0 to n - 1 and a door. Each person can enter or exit through the door once, taking one second.
    You are given a non-decreasing integer array arrival of size n, where arrival[i] is the arrival time of the ith person at the door.
    You are also given an array state of size n, where state[i] is 0 if person i wants to enter through the door or 1 if they want to exit through the door.

    If two or more persons want to use the door at the same time, they follow the following rules:
    - If the door was not used in the previous second, then the person who wants to exit goes first.
    - If the door was used in the previous second for entering, the person who wants to enter goes first.
    - If the door was used in the previous second for exiting, the person who wants to exit goes first.
    - If multiple persons want to go in the same direction, the person with the smallest index goes first.

    Return an array answer of size n where answer[i] is the second at which the ith person crosses the door.

    Note that:
    - Only one person can cross the door at each second.
    - A person may arrive at the door and wait without entering or exiting to follow the mentioned rules.
    """

    def __init__(self, arrival: list[int], state: list[int]):
        """
        Initializes the Solution class with the arrival times and states of the persons.
        """
        self.arrival = arrival
        self.state = state
        
    def __repr__(self):
        return f"Solution(arrival={self.arrival}, state={self.state})"

    def timeTakenToCross(self) -> list[int]:
        n = len(self.arrival)
        answer = [-1] * n  # Initialize the result list with -1

        enter_queue = []  # Queue for persons who want to enter
        exit_queue = []   # Queue for persons who want to exit

        time = 0  # Current time unit
        last_direction = -1  # -1 indicates the door was not used in the previous second

        i = 0  # Index to track the next person to arrive

        while i < n or enter_queue or exit_queue:
            # Enqueue persons who have arrived at the current time
            while i < n and self.arrival[i] <= time:
                if self.state[i] == 0:
                    # Person wants to enter
                    enter_queue.append(i)
                    print(colored(f"Time {time + 1}: Person {i+1} arrives to Enter", 'cyan'))
                else:
                    # Person wants to exit
                    exit_queue.append(i)
                    print(colored(f"Time {time + 1}: Person {i+1} arrives to Exit", 'magenta'))
                i += 1

            # Determine who can cross the door at this time
            if exit_queue or enter_queue:
                if (last_direction == -1):
                    # Door was not used in the previous second
                    if exit_queue:
                        # Give priority to person wanting to exit
                        index = exit_queue.pop(0)
                        last_direction = 1  # Update last direction to Exit
                        print(colored(f"Time {time + 1}: Person {index+1} Exits through the door", 'green'))
                    else:
                        # No one wants to exit, person wanting to enter goes
                        index = enter_queue.pop(0)
                        last_direction = 0  # Update last direction to Enter
                        print(colored(f"Time {time + 1}: Person {index+1} Enters through the door", 'green'))
                elif last_direction == 1:
                    # Door was last used for exiting
                    if exit_queue:
                        # Continue exiting
                        index = exit_queue.pop(0)
                        print(colored(f"Time {time + 1}: Person {index+1} Exits through the door", 'green'))
                    elif enter_queue:
                        # No one wants to exit, switch to entering
                        index = enter_queue.pop(0)
                        last_direction = 0  # Change direction to Enter
                        print(colored(f"Time {time + 1}: No one to Exit, Person {index+1} Enters (Change direction)", 'yellow'))
                else:
                    # Door was last used for entering
                    if enter_queue:
                        # Continue entering
                        index = enter_queue.pop(0)
                        print(colored(f"Time {time + 1}: Person {index+1} Enters through the door", 'green'))
                    elif exit_queue:
                        # No one wants to enter, switch to exiting
                        index = exit_queue.pop(0)
                        last_direction = 1  # Change direction to Exit
                        print(colored(f"Time {time + 1}: No one to Enter, Person {index+1} Exits (Change direction)", 'yellow'))
                answer[index] = time  # Record the time when the person crosses
            else:
                # No one is waiting at the door
                last_direction = -1  # Reset last_direction
                print(colored(f"Time {time + 1}: No one is at the door", 'red'))

            time += 1  # Move to the next time unit

        return answer

    def timeTakenToCross_enhanced(self) -> list[int]:
        """
        Optimized solution for calculating the time taken for people to cross a door.

        This implementation improves the time complexity from O(T + n) to O(n) by using
        an event-driven simulation approach. Instead of iterating over every time unit,
        the algorithm advances time to the next event (arrival or crossing), processing
        events efficiently.

        Attributes
        ----------
        arrival : List[int]
            The arrival times of each person.
        state : List[int]
            The desired direction of each person (0 for entry, 1 for exit).
        """
        n = len(self.arrival)
        answer = [-1] * n  # Initialize the result list with -1

        enter_queue = deque()  # Queue for persons who want to enter
        exit_queue = deque()   # Queue for persons who want to exit

        current_time = 0  # Current time unit
        last_used_time = -1  # The last time the door was used
        last_direction = 1   # Assume the door was not used at time -1, so default to exit

        i = 0  # Index to track the next person to arrive

        while i < n or enter_queue or exit_queue:
            # Advance current_time to the earliest of next arrival time or next possible crossing time
            if (not enter_queue and not exit_queue) and i < n and self.arrival[i] > current_time:
                current_time = self.arrival[i]
                last_direction = 1  # Reset last_direction since door was idle

            # Enqueue persons who have arrived by current_time
            while i < n and self.arrival[i] <= current_time:
                if self.state[i] == 0:
                    enter_queue.append(i)
                    print(colored(f"Time {current_time + 1}: Person {i+1} arrives to Enter", 'cyan'))
                else:
                    exit_queue.append(i)
                    print(colored(f"Time {current_time + 1}: Person {i+1} arrives to Exit", 'magenta'))
                i += 1

            # Determine who can cross
            if exit_queue or enter_queue:
                if last_used_time != current_time - 1:
                    # Door was not used in the previous second
                    if exit_queue:
                        index = exit_queue.popleft()
                        last_direction = 1
                    else:
                        index = enter_queue.popleft()
                        last_direction = 0
                else:
                    if last_direction == 1:
                        if exit_queue:
                            index = exit_queue.popleft()
                        elif enter_queue:
                            index = enter_queue.popleft()
                            last_direction = 0  # Change direction
                    else:
                        if enter_queue:
                            index = enter_queue.popleft()
                        elif exit_queue:
                            index = exit_queue.popleft()
                            last_direction = 1  # Change direction

                answer[index] = current_time
                print(colored(f"Time {current_time + 1}: Person {index+1} {'Exits' if last_direction == 1 else 'Enters'} through the door", 'green'))
                last_used_time = current_time
            else:
                # No one is waiting; reset last_direction
                last_direction = 1  # Default to exit if someone arrives next
                print(colored(f"Time {current_time + 1}: No one is at the door", 'red'))

            # Move to the next event
            current_time += 1

        return answer
    
def generate_test_case(n):
        arrival = sorted(random.randint(0, n//2) for _ in range(n))
        state = [random.randint(0, 1) for _ in range(n)]
        return arrival, state
    
if __name__ == "__main__":

    solution = Solution([0,1,1,2,4], [0,1,0,0,1])
    print(colored(f"{solution}", 'magenta'))
    print(colored('-' * 100, 'red'))

    start_time = time.time()
    result = solution.timeTakenToCross()
    end_time = time.time()
    print(colored(f"\nFinal crossing times: {result}", 'blue'))
    print(colored(f"Time taken for timeTakenToCross: {end_time - start_time} seconds", 'yellow'))
    print(colored('-' * 100, 'red'))

    start_time = time.time()
    result = solution.timeTakenToCross_enhanced()
    end_time = time.time()
    
    print(colored('-' * 100, 'red'))
    print(colored(f"\nFinal crossing times: {result}", 'blue'))
    print(colored(f"Time taken for timeTakenToCross_enhanced: {end_time - start_time} seconds", 'yellow'))
    print(colored('-' * 100, 'red'))
    
 
    # n = 10000  # Number of persons
    # arrival, state = generate_test_case(n)

    # # Original Approach
    # print(colored("Running Original Approach...", 'yellow'))
    # original_solution = Solution(arrival, state)
    # start_time = time.time()
    # original_result = original_solution.timeTakenToCross()
    # original_time = time.time() - start_time
    # print(colored(f"Original Approach Time: {original_time:.4f} seconds", 'blue'))

    # # Optimized Approach
    # print(colored("\nRunning Optimized Approach...", 'yellow'))
    # optimized_solution = Solution(arrival, state)
    # start_time = time.time()
    # optimized_result = optimized_solution.timeTakenToCross()
    # optimized_time = time.time() - start_time
    # print(colored(f"Optimized Approach Time: {optimized_time:.4f} seconds", 'blue'))

    # # Verify that both results are the same
    # assert original_result == optimized_result, "Results do not match!"