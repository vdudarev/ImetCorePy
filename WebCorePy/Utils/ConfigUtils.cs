using Microsoft.Extensions.Configuration;
using System.IO;

namespace WebCorePy.Utils;

public static class ConfigUtils
{
    public static string GetConnectionString(this IConfiguration config) {
        return config?.GetConnectionString("InfDB") ?? string.Empty;
    }

    static string _storageRoot = null!;
    public static string GetStorageRoot(this IConfiguration config)
    {
        if (_storageRoot == null) {
            _storageRoot = config.GetValue<string>("PathToFileStorage") ?? string.Empty;
        }
        return _storageRoot;
    }

    public static string GetRelativeFolderNameTemporary(this IConfiguration config, int tenantId, int userId)
    {
        return $"/tenant{tenantId}/temp/user{userId}";
    }
    //public static string GetRelativeFolderNameForObject(this IConfiguration config, ObjectInfo obj) {
    //    if (obj == null) return string.Empty;
    //    return $"/tenant{obj.TenantId}/type{obj.TypeId}/object{obj.ObjectId}";
    //}

    public static string MapStorageFile(this IConfiguration config, string relativeFileName)
    {
        string root = config.GetStorageRoot();
        relativeFileName = relativeFileName.Replace('/', '\\');
        string retVal = Path.Combine(root, relativeFileName.TrimStart('\\'));
        return retVal;
    }


    public static bool StorageFileExists(this IConfiguration config, string relativeFileName)
    {
        string physicalFileName = config.MapStorageFile(relativeFileName);
        return File.Exists(physicalFileName);
    }


}
