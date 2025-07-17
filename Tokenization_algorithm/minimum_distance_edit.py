con_matrix = {
    ("a", "e"): 342,
    ("e", "a"): 388,
    ("i", "a"): 103,
    ("o", "a"): 91,
    ("i", "e"): 146,
    ("o", "e"): 116,
    ("a", "i"): 118,
    ("e", "i"): 89,
    ("m", "n"): 180
}

high_cost = 10

def sub_cost(x, y):
    if x == y:
        return 0
    freq_cost = con_matrix.get((x, y), 0)
    if freq_cost > 0:
        return 1 / freq_cost
    else:
        return high_cost    

insert_cost = 1
delete_cost = 1

def dp_edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i * delete_cost
    for j in range(n+1):
        dp[0][j] = j * insert_cost

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = sub_cost(str1[i-1], str2[j-1])
            dp[i][j] = min(
                dp[i-1][j] + delete_cost,   # deletion
                dp[i][j-1] + insert_cost,   # insertion
                dp[i-1][j-1] + cost         # substitution
            )

    edits = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + sub_cost(str1[i-1], str2[j-1]):
            if str1[i-1] != str2[j-1]:
                edits.append(f"Substitute '{str1[i-1]}'→'{str2[j-1]}'")
            else:
                edits.append(f"Match '{str1[i-1]}'")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + delete_cost:
            edits.append(f"Delete '{str1[i-1]}'")
            i -= 1
        else:
            edits.append(f"Insert '{str2[j-1]}'")
            j -= 1

    edits.reverse()

    return dp[m][n], edits

# Input & Output
word1 = input("Enter first word: ")
word2 = input("Enter goal word: ")

distance, edit_steps = dp_edit_distance(word1, word2)

print("\nWeighted Edit Distance:", distance)
print("Traceback Steps (clearly):")
for step in edit_steps:
    print("-", step)
