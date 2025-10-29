using Microsoft.Extensions.Configuration;

namespace ConfigHelpers;

public class ConfigHelpers
{
    /// <summary>
    /// returns ConnectionString by access to configuration (appsettings.json) and known hostHame
    /// Ultimately: it either returns ConnectionStrings:InfDB_<hostName> OR ConnectionStrings:InfDB (as default)
    /// </summary>
    /// <param name="configuration"></param>
    /// <param name="hostName">name of the host in request (mdi.matinf.pro)</param>
    /// <returns>ConnectionString</returns>
    public static string GetConnectionString(Microsoft.Extensions.Configuration.IConfiguration configuration, string hostName = null)
    {
        const string prefix = "InfDB";
        string retVal = null;

        if (!string.IsNullOrEmpty(hostName))
        {
            retVal = configuration?.GetConnectionString($"{prefix}_{hostName}");
            if (!string.IsNullOrEmpty(retVal))
                return retVal;
        }
        retVal = configuration?.GetConnectionString(prefix);
        return retVal;
    }

}