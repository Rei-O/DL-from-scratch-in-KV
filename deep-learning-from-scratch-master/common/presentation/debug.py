
isDebugMode = True

def debugprt(val, comment=None, isDebug=False):
    """
    デバッグ用に標準出力を行う。
    comment : val で標準出力する。
    グローバル変数isDebugAllにより出力有無を制御する。
    isDebugAll=Falseの際にisDebug=Trueとすることで個別に標準出力するを有効にすることが出来る。
    param : val:Any, comment=None:Any, isDebug=False:Boolean 
    return : None
    """
    def _debugprt(val, comment):
        if comment == None:
            print(val)
        else:
            print(f"{comment} : {val}")
    
    if isDebugMode: 
        _debugprt(val, comment)
    elif isDebug:
        _debugprt(val, comment)
    else:
        pass