using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;

namespace ConfigHelpers;

public class ConfigHttpHelpers
{
    /// <summary>
    /// Gets a host inbformatilon from current request
    /// </summary>
    /// <param name="httpContext">current HttpContext (Microsoft.AspNetCore.Http.IHttpContextAccessor)</param>
    /// <returns>HostString object</returns>
    public static HostString GetHostFromHttpContext(Microsoft.AspNetCore.Http.IHttpContextAccessor httpContext)
        => httpContext.HttpContext.Request.Host;


    /// <summary>
    /// Gets canonical hostName string to resolve in configuration (connection strings / tenants)
    /// </summary>
    /// <param name="configuration">access to configuration (appsettings.json) via Microsoft.Extensions.Configuration.IConfiguration</param>
    /// <param name="httpContext">current HttpContext (Microsoft.AspNetCore.Http.IHttpContextAccessor)</param>
    /// <returns>canonical hostName</returns>
    public static string GetHostByHttpContext(Microsoft.Extensions.Configuration.IConfiguration configuration, Microsoft.AspNetCore.Http.IHttpContextAccessor httpContext)
    {
        HostString host = GetHostFromHttpContext(httpContext);
        // return GetHostByHttpContextHardcoded(host); // manual for pre-release
        return GetHostByHttpContextWithConfiguration(configuration, host);
    }


    /// <summary>
    /// Gets canonical hostName string to resolve in configuration (connection strings / tenants)
    /// </summary>
    /// <param name="configuration">access to configuration (appsettings.json) via Microsoft.Extensions.Configuration.IConfiguration</param>
    /// <param name="host">current HostString object (describes current request URL)</param>
    /// <returns>canonical hostName</returns>
    protected static string GetHostByHttpContextWithConfiguration(Microsoft.Extensions.Configuration.IConfiguration configuration, HostString host) {
        // 1 priority: host + port
        string key = host.Host + ":" + host.Port;
        // string value = configuration.GetSection($"AppSettings:HostAliases:{key}")?.Value;
        string value = configuration[$"HostAliases:{key}"];
        if (!string.IsNullOrEmpty(value))
            return value;
        // 2 priority: host
        key = host.Host;
        value = configuration.GetSection($"HostAliases:{key}")?.Value;
        if (!string.IsNullOrEmpty(value))
            return value;
        // fallback (default)
        return host.Host;
    }


    /// <summary>
    /// for pre-release (hardcoded/dirty version of GetHostByHttpContextWithConfiguration)
    /// </summary>
    /// <param name="host">current HostString object (describes current request URL)</param>
    /// <returns>canonical hostName</returns>
    [Obsolete("Do not use it, use GetHostByHttpContextWithConfiguration", true)]
    public static string GetHostByHttpContextHardcoded(HostString host)
    {
        if (string.Compare(host.Host, "pub.mdi.ruhr-uni-bochum.de", ignoreCase: true) == 0)
        {
            return "pub.matinf.pro";
        }
        if (string.Compare(host.Host, "db.mdi.ruhr-uni-bochum.de", ignoreCase: true) == 0)
        {
            return "mdi.matinf.pro";
        }
        if (string.Compare(host.Host, "localhost", ignoreCase: true) == 0)
        {
            switch (host.Port)
            {    // Properties/launchSettings.json
                case 5107:
                case 7107:
                    return "demi.matinf.pro";
                case 5106:
                case 7106:
                    return "mdi.matinf.pro";
                case 5105:
                case 7105:
                    return "pub.matinf.pro";
                case 5104:
                case 7104:
                    return "crc247.mdi.ruhr-uni-bochum.de";
                case 5103:
                case 7103:
                    return "crc1625.mdi.ruhr-uni-bochum.de";
                case 5102:
                case 7102:
                    return "demo.mdi.ruhr-uni-bochum.de";
                case 5101:
                case 7101:
                    return "dim.mdi.ruhr-uni-bochum.de";
                case 5100:
                case 7100:
                default:
                    return "inf.mdi.ruhr-uni-bochum.de";
            }
        }
        return host.Host;
    }

}