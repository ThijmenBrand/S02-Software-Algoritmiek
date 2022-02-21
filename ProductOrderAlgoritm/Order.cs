using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ProductOrderAlgoritm
{
    internal class Order
    {
        List<Product> products = new List<Product>
        {
            new Product(Name: "Rijst", 1.50),
            new Product(Name: "Soep", 2.25),
            new Product(Name: "Oreo", 2.36),
            new Product(Name: "komkommer", 0.95),
            new Product(Name: "Avocado", 1.05),
            new Product(Name: "Miso-soep", 4.20),
            new Product(Name: "Zalm filet", 6.20),
            new Product(Name: "oven snacks", 3.99),
            new Product(Name: "casino brood", 0.99),
            new Product(Name: "hagelslag", 1.55),
            new Product(Name: "pindakaas", 2.49)
        };

        //Without recursion
        public double GiveMaximumPrice()
        {
            double highest = 0.00;

            products.ForEach(p =>
            {
                if (p.Price > highest)
                    highest = p.Price;
            });

            return highest;
        }

        //Without recursion
        public double GiveMaximumPriceWithRecursion()
        {
            //Yet to be implemented
            return 0.00;
        }

        public double GiveAveragePrice()
        {
            double sum = 0.00;
            int count = 0;

            products.ForEach(p =>
            {
                sum += p.Price;
            });

            return sum / products.Count;
        }

        public List<Product> GetAllProducts(double minimumPrice)
        {
            List<Product> returnProducts = new List<Product>();

            products.ForEach(p =>
            {
                if (p.Price >= minimumPrice)
                    returnProducts.Add(p);
            });

            return returnProducts;
        }

        public List<Product> SortProductsByPrice()
        {
            List<Product> Sorted = new List<Product>();
            List<Product> Unsorted = new List<Product>(products);

            for (int i = 0; i < (Unsorted.Count+Sorted.Count); i++)
            {
                int smallest = findSmallest(Unsorted);
                Sorted.Add(Unsorted[smallest]);
                Unsorted.RemoveAt(smallest);
            }

            return Sorted;
        }

        private int findSmallest(List<Product> pList)
        {
            double smallest = pList[0].Price;
            int smallestIndex = 0;
            for(int i = 1; i < pList.Count; i++)
            {
                if(pList[i].Price < smallest)
                {
                    smallest = pList[i].Price;
                    smallestIndex = i;
                }
            }
            return smallestIndex;
        }
    }
}
