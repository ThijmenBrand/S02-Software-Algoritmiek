namespace PoolseVlag;
    public class Program
    {
        static void Main(string[] args)
        {
            string[] colors = { "red", "white", "white", "red", "white", "red", "white", "white", "red" };

        string[] returnvalue = sortPoolseVlag(colors);

        foreach (string color in returnvalue)
        {
            Console.Write(color + " ");
        }
    }

        public static string[] sortPoolseVlag(string[] colors)
        {
            for (int i = 0; i < colors.Length; i++)
            {
                if (colors[i].ToLower() == "white")
                {
                    for (int j = i; j < colors.Length; j++)
                    {
                        if (colors[i].ToLower() != colors[j].ToLower())
                        {
                            string tmp = colors[i];
                            colors[i] = colors[j];
                            colors[j] = tmp;
                            break;
                        }
                    }
                }
            }

            return colors;
        }
    }
