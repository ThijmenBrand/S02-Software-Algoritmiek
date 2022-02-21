using ProductOrderAlgoritm;

Order order = new Order();
double minPrice = 1.50;

Console.WriteLine(order.GiveMaximumPrice());
Console.WriteLine(order.GiveAveragePrice());

Console.WriteLine($"Products with price above {minPrice}");
order.GetAllProducts(minPrice).ForEach(p =>
{
    Console.WriteLine($"Name: {p.Name} and Price {p.Price}");
});

Console.WriteLine($"List of ordered products");
order.SortProductsByPrice().ForEach(p =>
{
    Console.WriteLine($"Name: {p.Name} and Price {p.Price}");
});