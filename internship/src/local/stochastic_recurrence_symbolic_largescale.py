from sage.all import var, SR, expand

def compute_Yn_symbolic(max_N):
    # Declare u and v as symbolic variables
    u, v = var('u v')
    
    # Initialize matrices with Sage's Symbolic Ring (SR) zeros
    pmfs = [[SR(0)] * (max_N + 1) for _ in range(max_N + 1)]
    cdfs = [[SR(0)] * (max_N + 1) for _ in range(max_N + 1)]
    
    # Base cases: Y_0 = 0, Y_1 = 0
    pmfs[0][0] = SR(1)
    pmfs[1][0] = SR(1)
    
    for i in range(2):
        for k in range(max_N + 1):
            cdfs[i][k] = SR(1)

    termination_prob = 1 - u - v

    for n_target in range(2, max_N + 1):
        # n_target is n+1
        n = n_target - 1

        # The maximum value Y_{n+1} can take is n.
        for k in range(n_target): 
            prob_k = SR(0)
            
            # U_n is uniformly distributed over {0, ..., n-1}
            for i in range(n):
                idx1 = i
                idx2 = n - 1 - i
                
                # I: sum of independent copies + 1
                c_prob = SR(0)
                if k >= 1:
                    for j in range(k):
                        c_prob += pmfs[idx1][j] * pmfs[idx2][k - 1 - j]
                        
                # J: max of independent copies
                cdfs1_k = cdfs[idx1][k]
                cdfs2_k = cdfs[idx2][k]
                cdfs1_k_minus_1 = cdfs[idx1][k - 1] if k > 0 else SR(0)
                cdfs2_k_minus_1 = cdfs[idx2][k - 1] if k > 0 else SR(0)
                
                m_prob = (cdfs1_k * cdfs2_k) - (cdfs1_k_minus_1 * cdfs2_k_minus_1)
                
                prob_k += u * c_prob + v * m_prob
                
            t_prob = SR(1) if k == 0 else SR(0)
            
            # Divide by n for the uniform distribution expectation
            pmfs[n_target][k] = ((prob_k / n) + (termination_prob * t_prob)).expand()
            
        current_cdf = SR(0)
        for k in range(max_N + 1):
            if k < n_target: 
                current_cdf += pmfs[n_target][k]
            # Store the CDF in n_target, not n
            cdfs[n_target][k] = current_cdf.expand()

    expected_values = []
    for i in range(1, max_N + 1):
        # The expected value calculation correctly ranges up to the max bound
        expected_y = sum((k * pmfs[i][k] for k in range(i)), SR(0))
        expected_values.append(expand(expected_y))

    truncated_pmfs = []
    truncated_cdfs = []
    
    for i in range(max_N + 1):
        
        support_size = i + 1 
        truncated_pmfs.append(pmfs[i][:support_size])
        truncated_cdfs.append(cdfs[i][:support_size])

    return truncated_pmfs, truncated_cdfs, expected_values