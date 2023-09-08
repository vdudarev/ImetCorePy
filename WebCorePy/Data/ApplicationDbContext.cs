using IdentityManagerUI.Models;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace WebCorePy.Data;

/// <summary>
/// https://learn.microsoft.com/en-us/aspnet/core/security/authentication/customize-identity-model?view=aspnetcore-6.0
/// https://learn.microsoft.com/ru-ru/dotnet/api/microsoft.aspnetcore.identity.entityframeworkcore.identitydbcontext-8?view=aspnetcore-6.0
/// </summary>
public class ApplicationDbContext : IdentityDbContext<
    ApplicationUser, // TUser: IdentityUser<TKey>
    ApplicationRole, // TRole: IdentityRole<TKey>
    int, // TKey - Тип первичного ключа для пользователей и ролей.
    ApplicationUserClaim,   // TUserClaim: IdentityUserClaim<TKey>
    ApplicationUserRole,    // TUserRole: IdentityUserRole<TKey>
    ApplicationUserLogin,   // TUserLogin: IdentityUserLogin<TKey>
    ApplicationRoleClaim,   // TRoleClaim : IdentityRoleClaim<TKey>
    ApplicationUserToken    // TUserToken : IdentityUserToken<TKey>
>
{
    public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
        : base(options)
    {
    }

    protected override void OnModelCreating(ModelBuilder builder)
    {
        base.OnModelCreating(builder);

        // Each User can have many UserClaims
        builder.Entity<ApplicationUser>().HasMany(e => e.Claims).WithOne(e => e.User).HasForeignKey(uc => uc.UserId).IsRequired().OnDelete(DeleteBehavior.Cascade);
        // Each User can have many UserLogins
        builder.Entity<ApplicationUser>().HasMany(e => e.Logins).WithOne(e => e.User).HasForeignKey(ul => ul.UserId).IsRequired().OnDelete(DeleteBehavior.Cascade);
        // Each User can have many UserTokens
        builder.Entity<ApplicationUser>().HasMany(e => e.Tokens).WithOne(e => e.User).HasForeignKey(ut => ut.UserId).IsRequired().OnDelete(DeleteBehavior.Cascade);
        // Each User can have many entries in the UserRole join table
        builder.Entity<ApplicationUser>().HasMany(e => e.Roles).WithOne(e => e.User).HasForeignKey(ur => ur.UserId).IsRequired().OnDelete(DeleteBehavior.Cascade);

        // Each Role can have many entries in the UserRole join table
        builder.Entity<ApplicationRole>().HasMany(e => e.Roles).WithOne(e => e.Role).HasForeignKey(ur => ur.RoleId).IsRequired().OnDelete(DeleteBehavior.Cascade);
        // Each Role can have many associated RoleClaims
        builder.Entity<ApplicationRole>().HasMany(e => e.Claims).WithOne(e => e.Role).HasForeignKey(rc => rc.RoleId).IsRequired().OnDelete(DeleteBehavior.Cascade);
    }
}

/*
public class ApplicationDbContext : IdentityDbContext<ApplicationUser, ApplicationRole, int>
{
    public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
        : base(options)
    {
    }

    protected override void OnModelCreating(ModelBuilder builder)
    {
        base.OnModelCreating(builder);

        builder.Entity<ApplicationUser>().HasMany(p => p.Roles).WithOne().HasForeignKey(p => p.UserId).IsRequired().OnDelete(DeleteBehavior.Cascade);
        builder.Entity<ApplicationUser>().HasMany(e => e.Claims).WithOne().HasForeignKey(e => e.UserId).IsRequired().OnDelete(DeleteBehavior.Cascade);
        builder.Entity<ApplicationRole>().HasMany(r => r.Claims).WithOne().HasForeignKey(r => r.RoleId).IsRequired().OnDelete(DeleteBehavior.Cascade);

        // important: the default is nvarchar(450), which is 900 bytes. Combined with other columns this is over 900 bytes.
        // SQL Server has a max key length of 900 bytes for a clustered index, which the default pushes over.
        //builder.Entity<ApplicationUser>().Property(u => u.Id).HasMaxLength(128);
        //builder.Entity<ApplicationRole>().Property(u => u.Id).HasMaxLength(128);
    }
}
*/
