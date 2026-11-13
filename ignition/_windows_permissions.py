# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner and SYSTEM access for private signing files on Windows."""

import ctypes as C


def _api():
    security = C.WinDLL("advapi32", use_last_error=True)
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        C.c_wchar_p, C.c_uint32, C.POINTER(C.c_void_p), C.c_void_p)
    security.SetFileSecurityW.argtypes = (C.c_wchar_p, C.c_uint32, C.c_void_p)
    security.GetNamedSecurityInfoW.argtypes = (
        C.c_wchar_p, C.c_int, C.c_uint32, C.POINTER(C.c_void_p), C.c_void_p,
        C.POINTER(C.c_void_p), C.c_void_p, C.POINTER(C.c_void_p))
    security.GetNamedSecurityInfoW.restype = C.c_uint32
    security.GetAce.argtypes = (C.c_void_p, C.c_uint32, C.POINTER(C.c_void_p))
    security.EqualSid.argtypes = (C.c_void_p, C.c_void_p)
    security.ConvertStringSidToSidW.argtypes = (C.c_wchar_p, C.POINTER(C.c_void_p))
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = (C.c_void_p,)
    kernel.LocalFree.restype = C.c_void_p
    return security, kernel


def restrict_file(path):
    security, kernel = _api()
    descriptor = C.c_void_p()
    if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            "D:P(A;;FA;;;SY)(A;;FA;;;OW)", 1, C.byref(descriptor), None):
        raise C.WinError(C.get_last_error())
    try:
        if not security.SetFileSecurityW(str(path), 0x80000004, descriptor):
            raise C.WinError(C.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


def owner_only(path):
    security, kernel = _api()
    owner, acl, descriptor = C.c_void_p(), C.c_void_p(), C.c_void_p()
    status = security.GetNamedSecurityInfoW(
        str(path), 1, 5, C.byref(owner), None, C.byref(acl), None, C.byref(descriptor))
    if status:
        raise C.WinError(status)
    allowed = []
    try:
        if not owner.value or not acl.value:
            return False
        for text in ("S-1-3-4", "S-1-5-18"):  # OWNER RIGHTS and SYSTEM.
            sid = C.c_void_p()
            if not security.ConvertStringSidToSidW(text, C.byref(sid)):
                raise C.WinError(C.get_last_error())
            allowed.append(sid)
        count = C.c_uint16.from_address(acl.value + 4).value
        for index in range(count):
            ace = C.c_void_p()
            if not security.GetAce(acl, index, C.byref(ace)):
                raise C.WinError(C.get_last_error())
            kind = C.c_ubyte.from_address(ace.value).value
            if kind == 1:  # Access-denied entries cannot grant access.
                continue
            if kind != 0 or not any(security.EqualSid(ace.value + 8, sid) for sid in (owner, *allowed)):
                return False
        return True
    finally:
        for sid in allowed:
            kernel.LocalFree(sid)
        kernel.LocalFree(descriptor)
