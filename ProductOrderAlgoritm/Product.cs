using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ProductOrderAlgoritm
{
    internal class Product
    {
        public Product(string Name, double Price)
        {
            this.Name = Name;
            this.Price = Price;
        }
        public string Name { get; set; }
        public double Price { get; set; }
    }
}
