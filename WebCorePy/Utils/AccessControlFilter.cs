namespace WebCorePy.Utils
{
    public class AccessControlFilter
    {
        /// <summary>
        /// By default public access only (for anonymous users)
        /// </summary>
        public AccessControl AccessControl { get; set; } = AccessControl.Public;

        /// <summary>
        /// if user is authorised, then UserId>0
        /// </summary>
        public int UserId { get; set; } = 0;


        public AccessControlFilter(AccessControl accessControl = AccessControl.Public, int userId = 0)
        {
            AccessControl = accessControl;
            UserId = userId;
        }

        public bool IsPublic => AccessControl == AccessControl.Public;
        public bool IsNone => AccessControl == AccessControl.None;

        public override string ToString() => $"AccessControlFilter: {AccessControl}, userId={UserId}";
    }

}
