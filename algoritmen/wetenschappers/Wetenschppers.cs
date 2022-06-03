using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace wetenschappers
{
    public class Wetenschappers
    {
        public static List<WetenschapperShape> parseWetenschappers()
        {
            string[] lines = System.IO.File.ReadAllLines(@"C:\Users\Thijmen\OneDrive\Fontys\S02\s02-software-algoritmiek\algoritmen\wetenschappers/wetenschappers.txt");
            List<WetenschapperShape> wetenschappers = new List<WetenschapperShape>();

            foreach (string line in lines)
            {
                string[] splitLine = line.Split(";");

                wetenschappers.Add(new WetenschapperShape(splitLine[0], int.Parse(splitLine[1]), int.Parse(splitLine[2])));
            }
            return wetenschappers;
        }
    }
}
