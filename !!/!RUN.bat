set PATH=C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3;%PATH%
set root=C:\ProgramData\Anaconda3
call %root%\Scripts\activate base
%root%\python.exe "D:\WWW\IMET\IMETCorePy\!!\regressor.py" "D:\WWW\IMET\IMETCorePy\!!\Upload\!settings.json"
call %root%\Scripts\deactivate.bat