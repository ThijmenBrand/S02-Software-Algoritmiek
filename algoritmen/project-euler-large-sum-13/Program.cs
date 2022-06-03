//Long way
List<int> multiples = new List<int>();

for (int i = 0; i < 1000; i++)
{
    int num = i * 3;
    int num_two = i * 5;

    multiples.Add(num);
    multiples.Add(num_two);
}

int outcome = 0;

foreach(int num in multiples)
{
    outcome += num;
}

Console.WriteLine(outcome);

//Better way
int result = 0;
for(int i = 0; i < 1000; i++)
{
    result += (i * 3) + (i * 5);
}

Console.WriteLine(result);
