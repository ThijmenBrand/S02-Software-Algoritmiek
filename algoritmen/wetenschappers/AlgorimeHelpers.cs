using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace wetenschappers
{
    public class AlgorimeHelpers
    {
        public int Jaar { get; set; }
        public int Aantal { get; set; }

        public AlgorimeHelpers(int jaar, int aantal)
        {
            Jaar = jaar;
            Aantal = aantal;
        }
    }
}
