using System;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.ViewEngines;
using System.Data.SqlClient;
using System.Data;
using System.Diagnostics;
using WebCorePy.Models;
using WebCorePy.Utils;
using Dapper;
using WebCorePy.DBContext;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity.UI.Services;
using Microsoft.AspNetCore.Identity;
using IdentityManagerUI.Models;
using System.Text;
using Azure.Core.GeoJson;
using System.Security.Cryptography;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace WebCorePy.Controllers;

[Authorize(Roles = "Administrator")]
public class SrvController : Controller
{
    private readonly UserManager<ApplicationUser> userManager;
    private readonly ILogger<SrvController> logger;
    private readonly IConfiguration config;
    private readonly DataContext dataContext;
    private readonly IWebHostEnvironment webHostEnvironment;
    private readonly IEmailSender mailSender;
    private readonly SmtpConfiguration smtpConfig;

    public SrvController(UserManager<ApplicationUser> userManager, ILogger<SrvController> logger, IConfiguration config, DataContext dataContext, IWebHostEnvironment webHostEnvironment, IEmailSender mailSender, SmtpConfiguration smtpConfig)
    {
        this.userManager = userManager;
        this.logger = logger;
        this.config = config;
        this.dataContext = dataContext;
        this.webHostEnvironment = webHostEnvironment;
        this.mailSender = mailSender;
        this.smtpConfig = smtpConfig;
    }

    public IActionResult Index() {
        var obj = new { First = "first", Second = "second" };
        var obj2 = new { First2 = "first22", Second2 = "second22" };
        logger.LogInformation("We have to write {obj} and later we write {obj2} !", obj, obj2);
        return View((webHostEnvironment, config, mailSender, smtpConfig, userManager));
    }


    /// <summary>
    /// Sends email
    /// </summary>
    /// <param name="email"></param>
    /// <param name="subject"></param>
    /// <param name="htmlMessage"></param>
    /// <returns></returns>
    [HttpPost]
    public async Task<JsonResult> SendMail(string email, string subject, string htmlMessage) {
        try
        {
            await mailSender.SendEmailAsync(email, subject, htmlMessage);
        }
        catch (Exception ex)
        {
            return Json(new { status = ex.GetType().ToString(), message = ex.Message });
        }
        return Json(new { status = "ok", message = "" });
    }

}