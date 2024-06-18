# Vitis-AI-API_2
Using Vitis-AI with KV260 board.
# Steps
<ubuntu>
-激活vitis ai 環境
-量化pytorch模型成xmodel
-用 "vai_c_xir -x /PATH/TO/quantized.xmodel -a /PATH/TO/arch.json -o /OUTPUTPATH -n netname" 指令來編譯xmodel

-再開一個終端機source petalinux
-寫好cpp
-用build.sh編譯cpp成執行檔

<windows>
-連線上板 放入兩個檔案
