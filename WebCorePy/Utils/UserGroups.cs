using System;

namespace WebCorePy.Utils;

[Flags]
public enum UserGroups {
    None = 0,
    User = 0x1,
    PowerUser = 0x2,
    Administrator = 0x4
} 
