def BYTE0(x):
    return x & 0xFF          # 取最低有效字节

def BYTE1(x):
    return (x >> 8) & 0xFF   # 取次低有效字节

def BYTE2(x):
    return (x >> 16) & 0xFF  # 取次高有效字节

def BYTE3(x):
    return (x >> 24) & 0xFF  # 取最高有效字节