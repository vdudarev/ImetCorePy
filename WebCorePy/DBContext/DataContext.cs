using Microsoft.AspNetCore.Mvc;
using System.Data.SqlClient;
using System.Data;
using Dapper;
using System.Security.Claims;
using System.Configuration;

namespace WebCorePy.DBContext;

public partial class DataContext
{
    #region General

    private static object myLock = new object();


    /// <summary>
    /// Database Connection String
    /// </summary>
    public string ConnectionString { get; init; }

    /// <summary>
    /// creates datacontext
    /// </summary>
    /// <param name="conn">ConnectionString</param>
    /// <param name="hostname">hostname (without https://)</param>
    public DataContext(string? conn)
    {
        ConnectionString = conn ?? string.Empty;
    }

    #endregion // General

}
