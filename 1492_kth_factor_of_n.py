import time
import tracemalloc
from termcolor import colored

class Solution:
    """
    1492. The kth Factor of n - MEDIUM

    You are given two positive integers n and k. A factor of an integer n is defined as an integer i where n % i == 0.

    Consider a list of all factors of n sorted in ascending order, return the kth factor in this list or return -1 if n has less than k factors.

    Example 1:

    Input: n = 12, k = 3
    Output: 3
    Explanation: Factors list is [1, 2, 3, 4, 6, 12], the 3rd factor is 3.
    Example 2:

    Input: n = 7, k = 2
    Output: 7
    Explanation: Factors list is [1, 7], the 2nd factor is 7.
    Example 3:

    Input: n = 4, k = 4
    Output: -1
    Explanation: Factors list is [1, 2, 4], there is only 3 factors. We should return -1.

    Constraints:

    1 <= k <= n <= 1000

    Follow up:

    Could you solve this problem in less than O(n) complexity?
    """

    def __init__(self, n: int, k: int):
        """
        Initializes the Solution class with n and k.
        """
        self.n = n
        self.k = k

    def __repr__(self):
        """
        Returns a string representation of the Solution class.
        """
        return f'Solution(n={self.n}, k={self.k})'

    def kthFactor(self) -> int:
        """
        Finds the kth factor of n.

        Returns
        -------
        int
            The kth factor of n or -1 if n has less than k factors.
        """
        # Brute force approach
        start_time = time.time()
        tracemalloc.start()
        
        kth_factor = []
        # Iterate through all numbers from 1 to n
        for i in range(1, self.n + 1):
            # Check if i is a factor of n
            if self.n % i == 0:
                # If it is, add it to the list of factors
                kth_factor.append(i)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(colored(f"Brute force approach took {end_time - start_time:.6f} seconds", 'blue'))
        print(colored(f"Memory usage: {current / 10**6:.6f} MB, Peak: {peak / 10**6:.6f} MB", 'blue'))
        
        # Return the kth factor if it exists, otherwise return -1
        return kth_factor[self.k - 1] if self.k <= len(kth_factor) else -1
    
    def kthFactor_optimized(self) -> int:
        """
        Finds the kth factor of n using an optimized approach.

        Returns
        -------
        int
            The kth factor of n or -1 if n has less than k factors.
        """
        start_time = time.time()
        tracemalloc.start()
        
        # List to store factors
        factors = []
        
        # Iterate through numbers from 1 to sqrt(n)
        for i in range(1, int(self.n**0.5) + 1):
            if self.n % i == 0:
                # If i is a factor of n, add it to the list of factors
                factors.append(i)
                # Check if i is not the square root of n
                # This ensures that we do not add the square root twice
                if i != self.n // i:
                    # If i is not the square root, add the corresponding factor n // i
                    factors.append(self.n // i)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(colored(f"Optimized approach took {end_time - start_time:.6f} seconds", 'green'))
        print(colored(f"Memory usage: {current / 10**6:.6f} MB, Peak: {peak / 10**6:.6f} MB", 'green'))
        
        # Return the kth factor if it exists, otherwise return -1
        return sorted(factors)[self.k - 1] if self.k <= len(factors) else -1

if __name__ == "__main__":
    solution = Solution(12, 3)
    print(colored(f"{repr(solution)}", 'red'))
    print(colored('-' * 100, 'red'))
    print(colored(f"Brute force approach solution: {solution.kthFactor()}", 'yellow'), 
          colored(f"n=12, k=3, space complexity: O(n), time complexity: O(n)", 'blue'))
    solution = Solution(7, 2)
    print(colored(f"{repr(solution)}", 'red'))
    print(colored('-' * 100, 'red'))
    print(colored(f"Brute force approach solution: {solution.kthFactor()}", 'yellow'), 
          colored(f"n=7, k=2, space complexity: O(n), time complexity: O(n)", 'blue'))
    solution = Solution(4, 4)
    print(colored(f"{repr(solution)}", 'red'))
    print(colored('-' * 100, 'red'))
    print(colored(f"Brute force approach solution: {solution.kthFactor()}", 'yellow'), 
          colored(f"n=4, k=4, space complexity: O(n), time complexity: O(n)", 'blue'))
    solution = Solution(12, 3)
    print(colored(f"{repr(solution)}", 'red'))
    print(colored('-' * 100, 'red'))
    print(colored(f"Optimized approach solution: {solution.kthFactor_optimized()}", 'yellow'), 
          colored(f"n=12, k=3, space complexity: O(sqrt(n)), time complexity: O(sqrt(n))", 'blue'))
    solution = Solution(7, 2)
    print(colored(f"{repr(solution)}", 'red'))
    print(colored('-' * 100, 'red'))
    print(colored(f"Optimized approach solution: {solution.kthFactor_optimized()}", 'yellow'), 
          colored(f"n=7, k=2, space complexity: O(sqrt(n)), time complexity: O(sqrt(n))", 'blue'))
    solution = Solution(4, 4)
    print(colored(f"{repr(solution)}", 'red'))
    print(colored('-' * 100, 'red'))
    print(colored(f"Optimized approach solution: {solution.kthFactor_optimized()}", 'yellow'), 
          colored(f"n=4, k=4, space complexity: O(sqrt(n)), time complexity: O(sqrt(n))", 'blue'))
