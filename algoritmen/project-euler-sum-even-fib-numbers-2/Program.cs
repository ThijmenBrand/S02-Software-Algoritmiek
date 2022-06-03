List<long> numbers = new List<long>() { 1,2 };

long thirdNum;
long sumOfEven = 0;

do 
{ 
    thirdNum = numbers[numbers.Count-1] + numbers[numbers.Count-2];
    numbers.Add(thirdNum);

    if (thirdNum % 2 == 0)
    {
        sumOfEven += thirdNum;
    }

} while (thirdNum <= 4000000000);

Console.WriteLine(sumOfEven);