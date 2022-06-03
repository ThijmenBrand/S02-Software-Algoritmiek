int tankVol(int h, int d, int vt)
{
    double r = d / 2;
    double TankLength = vt / (Math.PI * Math.Pow(r, 2));
    double TankSegment = Math.Pow(r, 2) * Math.Acos((r - h) / r) - (r - h) * Math.Sqrt(2 * r * h - Math.Pow(h, 2));

    return (int)Math.Floor(TankLength * TankSegment);
}

Console.WriteLine(tankVol(5, 7, 3848));