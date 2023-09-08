namespace WebCorePy.Utils
{
    /// <summary>
    /// настройки почтовой компоненты
    /// </summary>
    public class SmtpConfiguration
    {
        public string Host { get; set; } = "localhost";
        public int Port { get; set; } = 25;

        public string FromName { get; set; } = string.Empty;
        public string FromEmail { get; set; } = "noreply@imet-db.ru";

        public bool UseSSL { get; set; } = false;
        public string? CredentialsUserName { get; set; }
        public string? CredentialsPassword { get; set; }
        public string? CredentialsDomain { get; set; }
    }
}
