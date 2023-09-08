using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Google;
using Microsoft.AspNetCore.Authentication.OAuth;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Identity;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Claims;
using System.Threading.Tasks;

namespace WebCorePy.Utils;

/// <summary>
/// https://learn.microsoft.com/en-us/aspnet/core/security/authentication/social/google-logins?view=aspnetcore-6.0
/// https://console.cloud.google.com/apis/dashboard?project=materialsproject
/// </summary>
public class AuthGoogle
{
    public static void GoogleConfigureOptions(GoogleOptions googleOptions, WebApplicationBuilder builder) {
        static string? GetIdentityClaimFromProvider(OAuthCreatingTicketContext context, string propertyName, string? claimName = null)
        {
            string? retVal = null;
            if (string.IsNullOrEmpty(claimName))
                claimName = propertyName;
            try
            {
                retVal = context.User.GetProperty(propertyName).GetString();
                if (retVal != null) {
                    context.Identity?.AddClaim(new Claim(claimName, retVal));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
            return retVal;
        }


        static void RenewClaim(UserManager<IdentityManagerUI.Models.ApplicationUser> userManager, IdentityManagerUI.Models.ApplicationUser user, IList<Claim> claims, string claimName, string? claimValue)
        {
            Claim? curClaim = claims.FirstOrDefault(x => x?.Type == claimName);
            if (string.IsNullOrEmpty(curClaim?.Type) && !string.IsNullOrEmpty(claimValue))
            {
                userManager.AddClaimAsync(user, new Claim(claimName, claimValue));
            }
            else if (curClaim?.Type == claimName && !string.IsNullOrEmpty(claimValue) && curClaim.Value != claimValue)
            {
                userManager.ReplaceClaimAsync(user, curClaim, new Claim(claimName, claimValue));
            }
        }

        googleOptions.ClientId = builder.Configuration["Authentication:Google:ClientId"] ?? string.Empty;
        googleOptions.ClientSecret = builder.Configuration["Authentication:Google:ClientSecret"] ?? string.Empty;
        // https://learn.microsoft.com/en-us/aspnet/core/security/authentication/social/additional-claims?view=aspnetcore-6.0
        // googleOptions.ClaimActions.MapJsonKey("urn:google:picture", "picture", "url");
        // googleOptions.ClaimActions.MapJsonKey("urn:google:locale", "locale", "string");
        // https://stackoverflow.com/questions/45855503/how-to-retrieve-google-profile-picture-from-logged-in-user-with-asp-net-core-ide
        googleOptions.Scope.Add("email");
        googleOptions.Scope.Add("profile");
        googleOptions.Scope.Add("openid");
        //googleOptions.SaveTokens = true;
        googleOptions.Events.OnCreatingTicket = (context) =>
        {
            //{
            //  "id": "117346421118620353373",
            //  "email": "vic.dudarev@gmail.com",
            //  "verified_email": true,
            //  "name": "Victor Dudarev",
            //  "given_name": "Victor",
            //  "family_name": "Dudarev",
            //  "picture": "https://lh3.googleusercontent.com/a/ALm5wu2uts_praiLonIuGVMgoWSSr1MjG_FGWXFkDFphsw=s96-c",
            //  "locale": "ru"
            //}
            string json = context.User.ToString();

            var id = GetIdentityClaimFromProvider(context, "id");
            var email = GetIdentityClaimFromProvider(context, "email");
            var name = GetIdentityClaimFromProvider(context, "name");
            var given_name = GetIdentityClaimFromProvider(context, "given_name");
            var family_name = GetIdentityClaimFromProvider(context, "family_name");
            var picture = GetIdentityClaimFromProvider(context, "picture");

            // System.InvalidOperationException: 'No service for type 'Microsoft.AspNetCore.Identity.UserManager`1[IdentityManagerUI.Models.ApplicationUser]' has been registered.'
            var userManager = builder.Services.BuildServiceProvider().GetRequiredService<UserManager<IdentityManagerUI.Models.ApplicationUser>>();
            var user = userManager.FindByEmailAsync(email ?? string.Empty).Result;
            if (user != null)
            {
                IList<Claim> claims = userManager.GetClaimsAsync(user).Result;
                RenewClaim(userManager, user, claims, ClaimTypes.Name, name);
                RenewClaim(userManager, user, claims, ClaimTypes.GivenName, given_name);
                RenewClaim(userManager, user, claims, ClaimTypes.Surname, family_name);
                RenewClaim(userManager, user, claims, ClaimTypes.UserData, picture);
            }

            //context.ClaimActions.MapJsonKey(ClaimTypes.NameIdentifier, "id");
            //o.ClaimActions.MapJsonKey(ClaimTypes.Name, "name");
            //o.ClaimActions.MapJsonKey(ClaimTypes.GivenName, "given_name");
            //o.ClaimActions.MapJsonKey(ClaimTypes.Surname, "family_name");
            //o.ClaimActions.MapJsonKey("urn:google:profile", "link");
            //o.ClaimActions.MapJsonKey(ClaimTypes.Email, "email");
            //o.ClaimActions.MapJsonKey("image", "picture");


            // do not remove that (or claims will not ve saved)
            List<AuthenticationToken> tokens = context.Properties.GetTokens().ToList();
            tokens.Add(new AuthenticationToken()
            {
                Name = "TicketCreated",
                Value = DateTime.UtcNow.ToString()
            });
            context.Properties.StoreTokens(tokens);



            return Task.CompletedTask;
        };
    }
}
