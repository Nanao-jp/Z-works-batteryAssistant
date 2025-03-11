"""
Windowsでのpwdモジュールの互換性を提供するモジュール
"""

class DummyPwdUser:
    """Unix系システムのpwd.struct_passwdをエミュレート"""
    def __init__(self):
        self.pw_name = "dummy_user"
        self.pw_passwd = "x"
        self.pw_uid = 1000
        self.pw_gid = 1000
        self.pw_gecos = "Dummy User"
        self.pw_dir = "C:\\Users\\dummy"
        self.pw_shell = "C:\\Windows\\System32\\cmd.exe"

class pwd:
    """Unix系システムのpwdモジュールをエミュレート"""
    @staticmethod
    def getpwuid(uid):
        return DummyPwdUser()
    
    @staticmethod
    def getpwnam(name):
        return DummyPwdUser() 