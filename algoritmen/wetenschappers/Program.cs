using System.IO;
using System.Linq;
using wetenschappers;

List<WetenschapperShape> wetenschappers = Wetenschappers.parseWetenschappers();

Console.WriteLine(JaarMeesteWetenschappersOverleden());

string JaarMeesteWetenschappersOverleden()
{
    List<AlgorimeHelpers> jaren = new List<AlgorimeHelpers>();

    foreach(WetenschapperShape wetenschapper in wetenschappers)
    {
        bool jaarBestaatAl = false;
        foreach(AlgorimeHelpers jaar in jaren)
        {
            if(jaar.Jaar == wetenschapper.sterfteJaar)
            {
                jaarBestaatAl = true;
                jaar.Aantal++;
                break;
            }
        }
        if (!jaarBestaatAl)
        {
            jaren.Add(new AlgorimeHelpers(wetenschapper.sterfteJaar, 1));
        }
    }

    AlgorimeHelpers hoogste = new AlgorimeHelpers(0000, 0);
    foreach(AlgorimeHelpers jaar in jaren)
    {
        if(jaar.Aantal > hoogste.Aantal)
        {
            hoogste = jaar;
        }
    }

    return $"Het jaar met de meeste sterfte gevallen is {hoogste.Jaar} met {hoogste.Aantal} sterfte gevallen";
}