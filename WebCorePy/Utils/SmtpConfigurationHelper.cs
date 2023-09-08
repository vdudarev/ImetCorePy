using Microsoft.AspNetCore.Identity.UI.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace WebCorePy.Utils
{
    public static class SmtpConfigurationHelper
    {
        public static void AddEmailSenders<TImplementation>(this IServiceCollection services, IConfiguration configuration)
            where TImplementation : class, IEmailSender
        {
            SmtpConfiguration? smtpConfiguration = configuration.GetSection("SmtpConfiguration").Get<SmtpConfiguration>();
            if (smtpConfiguration == null || string.IsNullOrWhiteSpace(smtpConfiguration.Host))
                return;
            services.AddSingleton(smtpConfiguration);
            services.AddTransient<IEmailSender, TImplementation>();
        }

    }
}
