 # EXP.NO.10-Simulation-of-Error-Detection-and-Correction-Algorithms
10.Simulation of Error Detecion and Correction Algorithms

# AIM
To simulate error detection and correction techniques, such as Parity Check and Hamming Code, and to understand their concepts using Python.

# SOFTWARE REQUIRED
Google Colab

# ALGORITHMS
1. Initialize Matrices: Create empty lists for the parity matrix (pb), identity matrix (Ik), message bits, and error vectors.
2. Input Matrix Dimensions: Ask the user for the number of parity bits (col) and message bits (row).
3. Build Parity Matrix: Accept row-wise input for the parity matrix and validate that each row contains the correct number of elements.
4. Construct Generator Matrix: Convert the parity matrix into a NumPy array and concatenate it with an identity matrix to form the generator matrix.
5. Generate Codewords: Compute all possible message bit combinations, multiply them with the generator matrix, and store the resulting codewords.
6. Calculate Hamming Weights: Determine the Hamming weight for each codeword and identify the minimum Hamming distance (excluding the all-zero case).
7. Build Parity Check Matrix: Compute the transpose of the parity matrix, append an identity matrix, and calculate its transpose (H^T) for error detection.
8. Receive & Validate Codeword: Accept user input for the received codeword and validate that its length matches n.
9. Compute Syndrome & Error Vector: Multiply the received codeword with H^T to obtain the syndrome, then compare it against H^T rows to locate and correct errors.
10. Correct the Codeword: Add the error vector to the received codeword (mod 2) to correct errors, then display the corrected codeword along with the syndrome matrix.

# PROGRAM

    import numpy as np

    pb = []           # Parity matrix rows
    Ik = []           # Identity matrix
    p = []
    m = []
    h_dis = []
    r_code = []
    err = []

    col = int(input("Enter the number of parity bits: "))
    row = int(input("Enter the number of message bits: "))

    print("\nEnter the parity matrix rows:")
    for i in range(row):
        p = list(map(int, input(f"Row {i+1}: ").split()))
        if len(p) != col:
            raise ValueError(f"Each row must have {col} elements.")
        pb.append(p)

    p_mat = np.array(pb, dtype=int)
    Ik = np.eye(row, dtype=int)
    
    g_mat = np.hstack((Ik, p_mat))

    k = g_mat.shape[0]
    n = g_mat.shape[1]

    m = np.array([[1 if (i >> (k - j - 1)) & 1 else 0 for j in range(k)] for i in range(2 ** k)])

    c = np.mod(np.dot(m, g_mat), 2)

    for i, row in enumerate(c):
        h_dis1 = np.sum(row)
        h_dis.append(h_dis1)
    h_mat = np.array(h_dis).reshape(-1, 1)
    d_min = np.min(h_dis[1:])

    p_t = p_mat.T
    h_check = np.hstack((p_t, np.eye(col, dtype=int)))
    ht = h_check.T  # Transpose of H

    print("\n**********")
    print("Generator Matrix [G = I | P]:")
    for row in g_mat:
        print(" ".join(map(str, row)))

    print("\n**********")
    print("Message Bits\tCodeword\tHamming Weight")
    for i in range(len(m)):
        msg_str = " ".join(map(str, m[i]))
        code_str = " ".join(map(str, c[i]))
        print(f"{msg_str}\t{code_str}\t\t{h_dis[i]}")

    print("\n**********")
    print(f"Minimum Hamming Distance: {d_min}")

    print("\n**********")
    print("Parity Check Matrix [H = P^T | I]:")
    for row in h_check:
        print(" ".join(map(str, row)))

    print("\n**********")
    print("Transpose of Parity Check Matrix (H^T):")
    for row in ht:
        print(" ".join(map(str, row)))

    rc = list(map(int, input("\nEnter the received codeword: ").split()))
    if len(rc) != n:
        raise ValueError("Received codeword length must match codeword length n.")
    r_c = np.array([rc])

    e = np.mod(np.dot(r_c, ht), 2).flatten()

    err = np.zeros(n, dtype=int)
    for i in range(n):
        if np.array_equal(e, ht[i]):
            err[i] = 1
            break

    print("\n**********")
    print("Syndrome:", " ".join(map(str, e)))
    print("Error vector:", " ".join(map(str, err)))

    corrected = (r_c.flatten() + err) % 2
    print("Corrected Codeword:", " ".join(map(str, corrected)))

    print("\n**********")
    print("Syndrome Matrix:")
    for i in range(n):
        s = ht[i]
        ev = np.eye(n, dtype=int)[i]
        print(f"{' '.join(map(str, s))}  {' '.join(map(str, ev))}")
        
    print("**********")
   
# OUTPUT
![image](https://github.com/user-attachments/assets/f636de22-3bb3-43c6-8862-78728b87a2ff)
![image](https://github.com/user-attachments/assets/2243ba4e-6df7-4501-a129-431d9101250a)


 
# RESULT / CONCLUSIONS
Thus, the simulation of error detection using Parity Check and correction using Hamming Code was succesfully carried out.
