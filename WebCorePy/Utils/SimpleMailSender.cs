using System.Net.Mail;
using System.Net;
using System.Threading.Tasks;
using System;

namespace WebCorePy.Utils
{
    public class SimpleMailSender : Microsoft.AspNetCore.Identity.UI.Services.IEmailSender
    {
        private readonly Microsoft.Extensions.Logging.ILogger _logger;
        public SmtpConfiguration Config { get; } // Set with Secret Manager.

        public SimpleMailSender(SmtpConfiguration config, Microsoft.Extensions.Logging.ILogger<SimpleMailSender> logger)
        {
            Config = config;   // optionsAccessor.Value;
            _logger = logger;
        }

        public async Task SendEmailAsync(string email, string subject, string htmlMessage)
        {
            using MailMessage message = new MailMessage();
            message.To.Add(new MailAddress(email));
            if (string.IsNullOrEmpty(Config.FromName))
            {
                message.From = new MailAddress(Config.FromEmail);
            }
            else
            {
                message.From = new MailAddress(Config.FromEmail, Config.FromName);
            }
            message.Subject = subject;
            message.IsBodyHtml = true;
            message.Body = htmlMessage;

            using SmtpClient client = new SmtpClient(Config.Host);
            if (Config.Port > 0)
            {
                client.Port = Config.Port;
            }
            if (Config.UseSSL)
            {
                client.EnableSsl = true;
            }
            if (string.IsNullOrEmpty(Config.CredentialsUserName))
            {
                client.Credentials = CredentialCache.DefaultNetworkCredentials;
            }
            else
            {
                if (string.IsNullOrEmpty(Config.CredentialsDomain))
                    client.Credentials = new NetworkCredential(Config.CredentialsUserName, Config.CredentialsPassword);
                else
                    client.Credentials = new NetworkCredential(Config.CredentialsUserName, Config.CredentialsPassword, Config.CredentialsDomain);
            }
            await client.SendMailAsync(message);
            var from = Config.FromEmail;
            // TODO: uncomment _logger?.Log(LogLevel.Information, "Email to {email} from {from} was sent successfully [{subject}]!", email, from, subject);
        }

        /// <summary>
        /// Sending an email message through a standard .Net component
        /// </summary>
        /// <param name="fromEmail">from Email</param>
        /// <param name="fromName">from Name</param>
        /// <param name="toEmail">to Email</param>
        /// <param name="toName">to Name</param>
        /// <param name="subject">subject</param>
        /// <param name="body">body</param>
        /// <param name="IsBodyHtml">is in HTML format (true - HTML, false - plain text)</param>
        [Obsolete]
        public static void Send(string fromEmail, string fromName, string toEmail, string toName,
            string subject, string body, bool IsBodyHtml)
        {
            using MailMessage message = new MailMessage();
            message.To.Add(new MailAddress(toEmail, toName));
            message.From = new MailAddress(fromEmail, fromName);
            message.Subject = subject;
            message.IsBodyHtml = IsBodyHtml;
            message.Body = body;

            using SmtpClient client = new SmtpClient("localhost");
            client.Credentials = CredentialCache.DefaultNetworkCredentials;
            client.Send(message);
        }

    }
}
