Console.WriteLine(fact(3));

int fact(int n)
{
    if (n <= 1) // base case
        return 1;
    else
        return n * fact(n - 1);
}
