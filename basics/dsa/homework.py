import random
import time
import functools
import matplotlib.pyplot as plt

"""
The following functions are answers to question 1, 2, 3, 7 and 11. The answers to other questions are shown when running the code.
"""

# Question 1
def identity():
  return ((1, 0), (0, 1))

# Question 2
def product(A, B):
  return ((A[0][0]*B[0][0] + A[0][1]*B[1][0],
           A[0][0]*B[0][1] + A[0][1]*B[1][1]),
          (A[1][0]*B[0][0] + A[1][1]*B[1][0],
           A[1][0]*B[0][1] + A[1][1]*B[1][1]))

# Question 3
def naive(u, A, B):
  prod = identity()
  for i in u:
    if i == 'A':
      prod = product(prod, A)
    else:
      prod = product(prod, B)

  return prod

def random_word(n):
  return ''.join(random.choice(['A', 'B']) for _ in range(n))

# Question 7
def divide_and_conquer(u, A, B):
  if len(u) == 0:
    return identity()
  elif len(u) == 1:
    return A if u[0] == 'A' else B
  else:
    mid = len(u) // 2
    left_prod = divide_and_conquer(u[:mid], A, B)
    right_prod = divide_and_conquer(u[mid:], A, B)
    return product(left_prod, right_prod)

# Question 11
@functools.lru_cache(maxsize=None)
def opt_divide_and_conquer(u, A, B):
  if len(u) == 0:
    return identity()
  elif len(u) == 1:
    return A if u[0] == 'A' else B
  else:
    mid = len(u) // 2
    left_prod = opt_divide_and_conquer(u[:mid], A, B)
    right_prod = opt_divide_and_conquer(u[mid:], A, B)
    return product(left_prod, right_prod)

# Helper function to measure execution time
def get_time(func):
  t = time.time()
  func()
  return time.time() - t

if __name__ == "__main__":
  A = ((1, 2), (3, 4))
  B = ((5, 6), (7, 8))
  
  # Question 4
  u = random_word(100000)
  t_naive = get_time(lambda: naive(u, A, B))
  print(f"Question 4: time taken by naive for n=100000: {t_naive:.6f} seconds")

  # Question 8
  t_divide_and_conquer = get_time(lambda: divide_and_conquer(u, A, B))
  print(f"Question 8: time taken by divide_and_conquer for n=100000: {t_divide_and_conquer:.6f} seconds")


  # Question 10 (it takes a few minutes to execute, so I included the figures in the report)

  # n_values = list(range(1000, 100001, 1000))
  # times_naive = []
  # times_divide_and_conquer = []

  # for n in n_values:
  #     u = random_word(n)
  #     times_naive.append(get_time(lambda: naive(u, A, B)))
  #     times_divide_and_conquer.append(get_time(lambda: divide_and_conquer(u, A, B)))

  # plt.plot(n_values, times_naive, label='Naive')
  # plt.plot(n_values, times_divide_and_conquer, label='Divide and Conquer')
  # plt.xlabel('Length of word')
  # plt.ylabel('Time (seconds)')
  # plt.title('Question 10: Naive vs Divide and Conquer')
  # plt.legend()
  # plt.savefig('question10.png')
  # plt.close()


  # Question 12
  # n_values = list(range(1000, 100001, 1000))
  # times_divide_and_conquer = []
  # times_opt_divide_and_conquer = []
  # for n in n_values:
  #     u = random_word(n)
  #     times_divide_and_conquer.append(get_time(lambda: divide_and_conquer(u, A, B)))
  #     opt_divide_and_conquer.cache_clear()
  #     times_opt_divide_and_conquer.append(get_time(lambda: opt_divide_and_conquer(u, A, B)))

  # plt.plot(n_values, times_divide_and_conquer, label='Divide and Conquer')
  # plt.plot(n_values, times_opt_divide_and_conquer, label='Optimized Divide and Conquer')
  # plt.title('Question 12: Divide and Conquer vs Optimized Divide and Conquer')
  # plt.xlabel('Length of word')
  # plt.ylabel('Time (seconds)')
  # plt.legend()
  # plt.savefig('question12.png')
  # plt.close()

