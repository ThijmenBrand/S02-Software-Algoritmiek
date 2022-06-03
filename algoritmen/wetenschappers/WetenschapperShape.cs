using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace wetenschappers
{
    public class WetenschapperShape
    {
        public string naam { get; set; }
        public int geboorteJaar { get; set; }
        public int sterfteJaar { get; set; }

        public WetenschapperShape(string naam, int geboorteJaar, int sterfteJaar)
        {
            this.naam = naam;
            this.geboorteJaar = geboorteJaar;
            this.sterfteJaar = sterfteJaar;
        }
    }
}
