using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc.Infrastructure;
using Microsoft.AspNetCore.Routing;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Serilog;
using WebCorePy.Data;
using WebCorePy.DBContext;
using WebCorePy.Utils;
using System.Configuration;
using Microsoft.EntityFrameworkCore;
using FormHelper;
//"applicationUrl": "http://localhost:2967",
//"sslPort": 44392      -- разобраться, почему работает только на этом порту и где его менять - пробовал https://stackoverflow.com/questions/32840634/visual-studio-2015-iisexpress-change-ssl-port - не зашло

//LocalizationUtils.SetCulture();
var builder = WebApplication.CreateBuilder(args);
/*
builder.WebHost.UseKestrel(so =>
{
    so.Limits.KeepAliveTimeout = TimeSpan.FromHours(1);
    so.Limits.MaxRequestBodySize = 1024 * 1024 * 1024;
});
*/

var loggerConfig = new LoggerConfiguration();
if (builder.Environment.IsDevelopment())
    loggerConfig.MinimumLevel.Information();
else
    loggerConfig.MinimumLevel.Warning();
//.ReadFrom.Configuration(builder.Configuration)
//.Enrich.FromLogContext()
var logger = loggerConfig.WriteTo.Console()
.WriteTo.File(Path.Combine(builder.Environment.ContentRootPath, "Logs\\log.txt"), rollingInterval: RollingInterval.Day)
.CreateLogger();
Log.Logger = logger;
Log.Information("The global logger has been configured");

builder.Host.UseSerilog();






// https://learn.microsoft.com/en-us/aspnet/core/security/authentication/social/google-logins?view=aspnetcore-6.0
// https://console.cloud.google.com/apis/credentials/consent?project=materialsproject
builder.Services.AddAuthentication().AddGoogle(options => AuthGoogle.GoogleConfigureOptions(options, builder));


builder.Services.Configure<RouteOptions>(options =>
    options.LowercaseUrls = true);

// Session


builder.Services.AddHttpContextAccessor();  //.TryAddSingleton<IHttpContextAccessor, HttpContextAccessor>();
builder.Services.TryAddSingleton<IActionContextAccessor, ActionContextAccessor>();
builder.Services.AddDistributedMemoryCache();
builder.Services.AddSession();

// https://github.com/dotnet/aspnetcore/issues/20369
builder.Services.Configure<IISServerOptions>(options => {
    options.MaxRequestBodySize = 1073741824; // 1 Gb
});
builder.Services.Configure<KestrelServerOptions>(options =>
{
    options.Limits.MaxRequestBodySize = 1073741824; // 1 Gb - if don't set default value is: 30 MB
});
builder.Services.Configure<FormOptions>(x =>
{
    x.ValueLengthLimit = 1073741824; // 1 Gb
    x.MultipartBodyLengthLimit = 1073741824; // 1 Gb - if don't set default value is: 128 MB
    x.MultipartHeadersLengthLimit = 1073741824; // 1 Gb
});

// validation: https://docs.fluentvalidation.net/en/latest/aspnet.html#getting-started
// builder.Services.AddScoped<IValidator<ObjectInfo>, ObjectInfoValidator>();
// builder.Services.AddValidatorsFromAssemblyContaining<ObjectInfoValidator>();

builder.Services.AddScoped<DataContext>(provider => new DataContext(
    builder.Configuration?.GetConnectionString("InfDB")));

// Add services to the container.
//var connectionString = builder.Configuration.GetConnectionString("IdentityDB");
//builder.Services.AddDbContext<ApplicationDbContext>(options =>
//    options.UseSqlServer(connectionString));
builder.Services.AddScoped<ApplicationDbContext>(provider => {
    var options = new DbContextOptionsBuilder<ApplicationDbContext>();
    //string hostName = DataContext.GetHostByHttpContext(provider.GetService<IHttpContextAccessor>());
    var connectionString = builder.Configuration?.GetConnectionString("InfDB");
    options.UseSqlServer(connectionString);
    return new ApplicationDbContext(options.Options);
});


// builder.Services.AddDatabaseDeveloperPageExceptionFilter();


// Add email senders which is currently setup for SendGrid and SMTP
builder.Services.AddEmailSenders<SimpleMailSender>(builder.Configuration);


builder.Services.AddAuthorization(options =>
{
    options.AddPolicy("RequireAdministratorRole", policy => policy.RequireRole("Administrator"));
    options.AddPolicy("RequirePowerUserRole", policy => policy.RequireRole("PowerUser"));
    options.AddPolicy("RequireUserRole", policy => policy.RequireRole("User"));
});

builder.Services.AddDefaultIdentity<IdentityManagerUI.Models.ApplicationUser>(options => options.SignIn.RequireConfirmedAccount = true)
    // https://learn.microsoft.com/en-us/aspnet/core/security/authorization/roles?view=aspnetcore-6.0
    .AddRoles<IdentityManagerUI.Models.ApplicationRole>()
    .AddEntityFrameworkStores<ApplicationDbContext>();

builder.Services.AddRazorPages();

builder.Services.AddControllersWithViews().AddFormHelper();


// StartUpUsers.CreateRolesAndUsers(builder.Services.BuildServiceProvider(), builder.Configuration["Authentication:Admin:Email"], builder.Configuration["Authentication:Admin:Password"]);    // run once only to feed DB

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    //app.UseMigrationsEndPoint();
    app.UseDeveloperExceptionPage();
}
else
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseStatusCodePagesWithReExecute("/StatusCode/{0}");

app.UseSession();
app.UseHttpsRedirection();
app.UseStaticFiles();


//app.UseWhen(
//    ctx => ctx.Request.Path.StartsWithSegments("/typevalidation"),
//    ab => ab.UseMiddleware<EnableRequestBodyBufferingMiddleware>()
//);

app.UseRouting();

app.UseAuthentication();
app.UseAuthorization();

app.MapControllerRoute(
    name: "areas",
    pattern: "{area}/{controller=Home}/{action=Index}/{id?}");
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");
app.MapRazorPages();
System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
app.UseFormHelper();
app.Run();
