using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Localization;
using Microsoft.AspNetCore.Mvc;

namespace WebCorePy.Controllers
{
    [AllowAnonymous]
    public class LanguageController : Controller
    {
        public IActionResult Index(string culture)
        {
            Response.Cookies.Append(
                CookieRequestCultureProvider.DefaultCookieName,
                CookieRequestCultureProvider.MakeCookieValue(new RequestCulture(culture)),
                new Microsoft.AspNetCore.Http.CookieOptions { Expires = System.DateTimeOffset.UtcNow.AddYears(1) }
            );
            string returnUrl = Request.Headers.Referer.ToString();
            return Redirect(returnUrl);
        }
    }
}
