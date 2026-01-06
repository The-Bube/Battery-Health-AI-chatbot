
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

memo = {}

#  public long fib2(int n){
#         if (n <= 1)
#             return n;
#         if (fib.containsKey(n)) {
#             return fib.get(n);
#         }

#         fib.put(n, fib2(n -1) + fib2(n-2));
#         return fib.get(n);
#     }
def fib_memo(n):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n  -1) + fib_memo(n -2)
    return  memo[n]

def fib_2(n):
    if n <= 1:
        return n
    a,b = 0,1
    for n in range(2, n + 1):
        a,b = b, a + b
    return b

for i in range(101):
    print(f"fib({i})={fib_2(i)}")